import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patchify the image into non-overlapping patches using Conv2d.

    Important:
      - DOES NOT add CLS token
      - DOES NOT add positional embedding
      - Outputs [B, N, D] where N=(H/patch)*(W/patch)
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size      # 96/8 = 12
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

        # Conv-based patchify
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self._init_weights()

    def _init_weights(self):
        # Better initialization for ViT patch projection
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.normal_(self.proj.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        # flatten
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        return x
