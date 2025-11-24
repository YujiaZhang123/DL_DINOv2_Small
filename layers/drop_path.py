import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        # keep each sample with probability (1 - drop_prob)
        keep_prob = 1.0 - self.drop_prob

        # generate random mask of shape [B, 1, 1, ...]
        batch_size = x.shape[0]
        mask_shape = (batch_size,) + (1,) * (x.ndim - 1)
        mask = torch.rand(mask_shape, dtype=x.dtype, device=x.device)

        # create binary gating mask
        mask = (mask < keep_prob).float() / keep_prob

        return x * mask
