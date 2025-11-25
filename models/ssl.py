import torch
import torch.nn as nn

from models.vision_transformer import VisionTransformer
from models.dino_head import DINOHead
from models.dino_loss import DINOLoss


class SSLArch(nn.Module):

    def __init__(
        self,
        img_size=96,
        patch_size=8,
        embed_dim=416,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_path_rate=0.03,
        num_prototypes=8192,
        n_global_crops=2,
        n_local_crops=8,
    ):
        super().__init__()

        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        # student backbone
        self.student_backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # teacher backbone
        self.teacher_backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=0.0,
        )

        self._init_teacher()

        # heads
        self.student_head = DINOHead(embed_dim, 2048, 256, num_prototypes)
        self.teacher_head = DINOHead(embed_dim, 2048, 256, num_prototypes)

        self._copy_student_to_teacher_head()

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # DINO loss
        self.dino_loss = DINOLoss(
            out_dim=num_prototypes,
            student_temp=0.1,
            teacher_temp=0.04,
            center_momentum=0.9,
        )

    def _init_teacher(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())

    def _copy_student_to_teacher_head(self):
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    @torch.no_grad()
    def update_teacher(self, momentum):
        for sp, tp in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            tp.data = tp.data * momentum + sp.data * (1 - momentum)

        for sp, tp in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            tp.data = tp.data * momentum + sp.data * (1 - momentum)


    def forward_teacher(self, global_crops):
        teacher_logits = []
        self.teacher_backbone.eval()
        self.teacher_head.eval()

        for g in global_crops:
            g_emb = self.teacher_backbone(g)        # [B, D]
            out = self.teacher_head(g_emb)
            teacher_logits.append(out["prototypes"])

        return teacher_logits

    def forward_student(self, global_crops, local_crops):
        student_logits = []

        for g in global_crops:
            g_emb = self.student_backbone(g)        # [B, D]
            out = self.student_head(g_emb)
            student_logits.append(out["prototypes"])

        for l in local_crops:
            l_emb = self.student_backbone(l)        # [B, D]
            out = self.student_head(l_emb)
            student_logits.append(out["prototypes"])

        return student_logits

    def forward(self, batch, teacher_temp=None):
        global_crops = batch["global_crops"]
        local_crops  = batch["local_crops"]

        # teacher outputs
        with torch.no_grad():
            teacher_logits = self.forward_teacher(global_crops)

        # student outputs
        student_logits = self.forward_student(global_crops, local_crops)

        if teacher_temp is not None:
            self.dino_loss.teacher_temp = teacher_temp

        # loss
        loss = self.dino_loss(student_logits, teacher_logits)
        logs = {"loss": loss.detach()}

        return loss, logs
