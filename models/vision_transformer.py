import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.patch_embed import PatchEmbed
from layers.block import TransformerBlock


class PatchNorm(nn.Module):
    """
    Patch Normalization used in DINOv2.
    Normalize each patch embedding across the embedding dimension.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        dropout_rate: float = 0.0,     
        use_patchnorm: bool = True,    
        use_layerscale: bool = True,  
    ):
        super().__init__()

        # -------- 1. Patch embedding --------
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches  # 96x96 -> 6×6=36 tokens

        # Optional PatchNorm
        self.use_patchnorm = use_patchnorm
        if use_patchnorm:
            self.patchnorm = PatchNorm(embed_dim)
        else:
            self.patchnorm = nn.Identity()

        # Patch dropout
        self.patch_dropout = nn.Dropout(dropout_rate)

        # -------- 2. CLS token + positional embedding --------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # -------- 3. DropPath schedule (stochastic depth) --------
        dpr_values = np.linspace(0, drop_path_rate, depth)

        # -------- 4. Transformer blocks --------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=float(dpr_values[i]),
                use_layerscale=use_layerscale,
            )
            for i in range(depth)
        ])

        # -------- 5. Final LN --------
        self.norm = nn.LayerNorm(embed_dim)

        # -------- 6. Initialize Parameters --------
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def get_pos_embed(self, n_patches: int, device=None):
        if n_patches + 1 == self.pos_embed.shape[1]:
            return self.pos_embed.to(device) if device is not None else self.pos_embed
            
        cls_pos = self.pos_embed[:, :1, :]    # [1,1,D]
        patch_pos = self.pos_embed[:, 1:, :]  # [1, base_N, D]

        base_n = patch_pos.shape[1]
        base_hw = int(math.sqrt(base_n))
        assert base_hw * base_hw == base_n, "base_n is not a square number"

        new_hw = int(math.sqrt(n_patches))
        assert new_hw * new_hw == n_patches, f"n_patches={n_patches} invalid"

        # [1, base_N, D] -> [1, D, base_hw, base_hw]
        patch_pos = patch_pos.reshape(1, base_hw, base_hw, -1).permute(0, 3, 1, 2)

        patch_pos = F.interpolate(
            patch_pos,
            size=(new_hw, new_hw),
            mode="bicubic",
            align_corners=False,
        )

        # [1, D, new_hw, new_hw] -> [1, new_N, D]
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, n_patches, -1)

        pos = torch.cat([cls_pos, patch_pos], dim=1)  # [1, n_patches+1, D]
        return pos.to(device) if device is not None else pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0)

        # 1. patchify -> [B, N, D]  (global: N=144, local: N=36)
        x = self.patch_embed(x)

        # 2. PatchNorm (optional)
        x = self.patchnorm(x)

        # 3. dropout
        x = self.patch_dropout(x)
        n_patches = x.size(1)

        # 4. Add CLS
        cls_tok = self.cls_token.expand(B, -1, -1)   # [B,1,D]
        x = torch.cat([cls_tok, x], dim=1)           # [B, N+1, D]

        # 5. Add (interpolated) positional embed
        pos = self.get_pos_embed(n_patches, device=x.device)  # [1, N+1, D]
        x = x + pos

        # 6. Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 7. Final norm + take CLS
        x = self.norm(x)
        return x[:, 0]
