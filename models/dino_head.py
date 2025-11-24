import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """
    3-layer MLP used in the DINO projection head:
        input_dim  → hidden_dim → hidden_dim → bottleneck_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()

        # 3-layer MLP: in → hidden → hidden → bottleneck
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, bottleneck_dim)

        self._init_weights()

    def _init_weights(self):
        # Use truncated normal style initialization
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.trunc_normal_(self.fc3.weight, std=0.02)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class DINOHead(nn.Module):
    """
    DINO-style projection head:
        MLP → bottleneck → normalize → weight-normalized prototypes
    """

    def __init__(
        self,
        input_dim: int = 480,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_prototypes: int = 8192,
    ):
        super().__init__()

        # Projection MLP (3 layers)
        self.mlp = ProjectionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # Prototypes (weight normalized)
        self.prototypes = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        )

        # Normalize weight norm scale (g parameter)
        with torch.no_grad():
            self.prototypes.weight_g.fill_(1.0)
            w = self.prototypes.weight_v.data
            self.prototypes.weight_v.copy_(F.normalize(w, dim=1, p=2))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, input_dim] CLS embeddings

        Returns:
            - projection: normalized bottleneck features
            - logits: prototype scores
        """
        # Step 1: projection MLP
        z = self.mlp(x)

        # Step 2: normalize features (very important for DINO!)
        z = F.normalize(z, p=2, dim=1)

        # Step 3: prototype logits
        logits = self.prototypes(z)

        return {
            "proj": z,
            "prototypes": logits,
        }
