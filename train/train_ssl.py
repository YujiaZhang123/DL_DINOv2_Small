import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.augmentations import DataAugmentationDINO
from data.dataset import SSLDataset
from data.collate import collate_multicrop
from data.loader import create_dataloader
from models.ssl import SSLArch


#################################################################
# Config
#################################################################
@dataclass
class TrainingConfig:
    # ----- data -----
    local_dir: str = "./hf_dataset"   # use local data
    split: str = "train"              # use train/ folder
    img_size: int = 96
    patch_size: int = 8              

    # ----- model -----
    embed_dim: int = 408
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_prototypes: int = 8192

    # ----- multi-crop -----
    n_global_crops: int = 2
    n_local_crops: int = 8
    global_crops_scale: tuple = (0.4, 1.0)
    local_crops_scale: tuple = (0.05, 0.3)

    # ----- optimization -----
    batch_size: int = 200             
    num_workers: int = 20
    epochs: int = 220
    base_lr: float = 2e-4
    min_lr: float = 2e-6
    weight_decay: float = 0.04
    warmup_epochs: int = 10

    momentum_teacher_base: float = 0.995
    momentum_teacher_final: float = 0.9995

    teacher_temp_warmup: float = 0.04
    teacher_temp_final: float = 0.07
    teacher_temp_warmup_epochs: int = 22

    device: str = "cuda"
    output_dir: str = "checkpoints"


#################################################################
# Seed
#################################################################
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#################################################################
# Load LOCAL DATASET (NO DOWNLOAD)
#################################################################
def load_local_dataset(cfg: TrainingConfig):

    data_root = os.path.join(cfg.local_dir, cfg.split)

    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"[ERROR] folder not found: {data_root}\n"
            f"upload files to {cfg.local_dir}/{cfg.split}/"
        )

    img_paths = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(root, f))

    print(f"[dataset] find {len(img_paths)} images in {data_root}")

    return img_paths


#################################################################
# Schedules
#################################################################
def cosine_schedule(base, final, total_steps, warmup_steps=0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return final / base + 0.5 * (1 - final / base) * (1 + math.cos(math.pi * t))
    return lr_lambda


def teacher_momentum_schedule(base, final, total_steps):
    def m_lambda(step):
        t = min(step / max(1, total_steps), 1.0)
        return base + (final - base) * (0.5 * (1 - math.cos(math.pi * t)))
    return m_lambda


def teacher_temp_schedule(cfg, global_step, steps_per_epoch):
    warmup_steps = cfg.teacher_temp_warmup_epochs * steps_per_epoch
    if global_step >= warmup_steps:
        return cfg.teacher_temp_final
    alpha = global_step / warmup_steps
    return cfg.teacher_temp_warmup + alpha * (cfg.teacher_temp_final - cfg.teacher_temp_warmup)


#################################################################
# Build dataloader
#################################################################
def build_dataloader(cfg):

    img_paths = load_local_dataset(cfg)

    augment = DataAugmentationDINO(
        global_crops_scale=cfg.global_crops_scale,
        local_crops_scale=cfg.local_crops_scale,
        local_crops_number=cfg.n_local_crops,
        global_crops_size=cfg.img_size,         # 96
        local_crops_size=cfg.img_size // 3,     # 32
    )

    dataset = SSLDataset(img_paths, augment)

    return create_dataloader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_multicrop,
    )


#################################################################
# Build model
#################################################################
def build_model(cfg):
    return SSLArch(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_path_rate=0.025,
        num_prototypes=cfg.num_prototypes,
        n_global_crops=cfg.n_global_crops,
        n_local_crops=cfg.n_local_crops,
    )


#################################################################
# Training
#################################################################
def train(cfg: TrainingConfig):
    set_seed()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataloader = build_dataloader(cfg)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * cfg.epochs

    model = build_model(cfg).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
    )

    lr_schedule = LambdaLR(
        optimizer,
        lr_lambda=cosine_schedule(
            base=cfg.base_lr,
            final=cfg.min_lr,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        ),
    )

    m_schedule = teacher_momentum_schedule(
        base=cfg.momentum_teacher_base,
        final=cfg.momentum_teacher_final,
        total_steps=total_steps,
    )

    global_step = 0
    print("==> Start training")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, ncols=120, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            global_crops = [x.to(device, non_blocking=True) for x in batch["global_crops"]]
            local_crops = [x.to(device, non_blocking=True) for x in batch["local_crops"]]

            data_dict = {
                "global_crops": global_crops,
                "local_crops": local_crops,
            }

            teacher_temp = teacher_temp_schedule(cfg, global_step, steps_per_epoch)
            momentum = m_schedule(global_step)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = model(data_dict, teacher_temp=teacher_temp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()

            with torch.no_grad():
                model.update_teacher(momentum)

            step_loss = loss.item()
            epoch_loss += step_loss

            pbar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "temp": f"{teacher_temp:.4f}",
                "m": f"{momentum:.5f}",
            })

            global_step += 1

        print(f"[epoch {epoch+1}] avg_loss = {epoch_loss/steps_per_epoch:.4f}")

        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epochs:
            ckpt = {
                "epoch": epoch + 1,
                "student_backbone": model.student_backbone.state_dict(),
            }
            ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"[saved] {ckpt_path}")


#################################################################
if __name__ == "__main__":
    cfg = TrainingConfig()
    train(cfg)
