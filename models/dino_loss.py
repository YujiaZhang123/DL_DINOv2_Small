import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    DINO loss, closer to the official implementation:
      - teacher only sees global crops
      - student sees global + local crops
      - global student views are matched to *other* teacher global views (i != j)
      - local student views are matched to all teacher global views
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9, 
    ):
        super().__init__()

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Running center for teacher logits
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor):
        """
        teacher_logits: [N_total_global_views * B, out_dim]
        """
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )

    @torch.no_grad()
    def teacher_distribution(self, logits: torch.Tensor):
        """
        Convert teacher logits to probability distribution with centering + temperature.
        logits: [B, out_dim]
        """
        centered = (logits - self.center) / self.teacher_temp
        return F.softmax(centered, dim=-1)

    def forward(self, student_logits_list, teacher_logits_list):
        """
        Args:
            student_logits_list: list of tensors, each [B, out_dim]
                - first n_global elements are student globals
                - remaining elements are student locals
            teacher_logits_list: list of tensors, each [B, out_dim]
                - global teacher outputs only

        Returns:
            scalar loss
        """
        # ----------------------------
        # 1) update
        # ----------------------------
        with torch.no_grad():
            all_teacher = torch.cat(teacher_logits_list, dim=0)
            self.update_center(all_teacher)

            teacher_probs = [self.teacher_distribution(t) for t in teacher_logits_list]

        n_global = len(teacher_logits_list)
        assert n_global >= 1, "Need at least one global crop for teacher."

        student_globals = student_logits_list[:n_global]
        student_locals = student_logits_list[n_global:]

        total_loss = 0.0
        n_loss_terms = 0

        # ----------------------------
        # 2) global student vs *other* global teacher views
        # ----------------------------
        for i, s in enumerate(student_globals):
            # s: [B, out_dim]
            student_logp = F.log_softmax(s / self.student_temp, dim=-1)
            for j, t in enumerate(teacher_probs):
                if i == j:
                    continue  
                loss = -(t * student_logp).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        # ----------------------------
        # 3) local student vs all global teacher views
        # ----------------------------
        for s in student_locals:
            student_logp = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_probs:
                loss = -(t * student_logp).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        return total_loss / max(n_loss_terms, 1)
