import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
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
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_prototypes: int = 8192

    # ----- multi-crop -----
    n_global_crops: int = 2
    n_local_crops: int = 6
    global_crops_scale: tuple = (0.4, 1.0)
    local_crops_scale: tuple = (0.1, 0.3)

    # ----- optimization -----
    batch_size: int = 150
    num_workers: int = 16
    epochs: int = 180
    base_lr: float = 2.5e-4
    min_lr: float = 2e-6
    weight_decay: float = 0.04
    warmup_epochs: int = 15

    momentum_teacher_base: float = 0.995
    momentum_teacher_final: float = 0.9995

    teacher_temp_warmup: float = 0.04
    teacher_temp_final: float = 0.07
    teacher_temp_warmup_epochs: int = 33

    device: str = "cuda"
    output_dir: str = "checkpoints"


#################################################################
# DDP utils
#################################################################
def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    return distributed, rank, world_size, local_rank


#################################################################
# Seed
#################################################################
def set_seed(seed=66, rank=0):
    import random
    import numpy as np
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


#################################################################
# Load LOCAL DATASET (NO DOWNLOAD)
#################################################################
def load_local_dataset(cfg: TrainingConfig, rank: int):

    data_root = os.path.join(cfg.local_dir, cfg.split)

    if not os.path.exists(data_root):
        if rank == 0:
            raise FileNotFoundError(
                f"[ERROR] folder not found: {data_root}\n"
                f"upload files to {cfg.local_dir}/{cfg.split}/"
            )
        else:
            return []

    img_paths = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(root, f))

    if rank == 0:
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
def build_dataloader(cfg, rank: int):

    img_paths = load_local_dataset(cfg, rank)

    augment = DataAugmentationDINO(
        global_crops_scale=cfg.global_crops_scale,
        local_crops_scale=cfg.local_crops_scale,
        local_crops_number=cfg.n_local_crops,
        global_crops_size=cfg.img_size,         # 96
        local_crops_size=cfg.img_size // 2,     # 48
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
        drop_path_rate=0.08,
        num_prototypes=cfg.num_prototypes,
        n_global_crops=cfg.n_global_crops,
        n_local_crops=cfg.n_local_crops,
    )


#################################################################
# Training
#################################################################
def train(cfg: TrainingConfig):
    # ----- init distributed -----
    distributed, rank, world_size, local_rank = init_distributed_mode()
    set_seed(42, rank)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    )

    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"[DDP] distributed={distributed}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"[DDP] device = {device}")

    dataloader = build_dataloader(cfg, rank)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * cfg.epochs

    # ----- build model -----
    model = build_model(cfg).to(device)

    # 只把 student 包进 DDP 也可以，这里图简单，直接包整个 SSLArch
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    model.train()

    # ----- optimizer & schedulers -----
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
    if is_main_process():
        print("==> Start training")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0

        if distributed and hasattr(dataloader, "sampler") and dataloader.sampler is not None:
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

        data_iter = dataloader
        if is_main_process():
            data_iter = tqdm(dataloader, ncols=120, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in data_iter:
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
                if isinstance(model, DDP):
                    model.module.update_teacher(momentum)
                else:
                    model.update_teacher(momentum)

            step_loss = loss.detach()
            epoch_loss += step_loss.item()

            if is_main_process():
                if isinstance(data_iter, tqdm):
                    data_iter.set_postfix({
                        "loss": f"{step_loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "temp": f"{teacher_temp:.4f}",
                        "m": f"{momentum:.5f}",
                    })

            global_step += 1

        if is_main_process():
            avg_loss = epoch_loss / steps_per_epoch
            print(f"[epoch {epoch+1}] avg_loss = {avg_loss:.4f}")

            if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epochs:
                arch = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch + 1,
                    "student_backbone": arch.student_backbone.state_dict(),
                }
                ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pth")
                torch.save(ckpt, ckpt_path)
                print(f"[saved] {ckpt_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


#################################################################
if __name__ == "__main__":
    cfg = TrainingConfig()
    train(cfg)
