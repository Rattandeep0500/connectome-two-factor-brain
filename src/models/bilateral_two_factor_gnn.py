import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class BilateralTwoFactorGNN(nn.Module):
    """
    Connectome model with:

    1. General factor pathway (g)
    2. Task-specific factor pathway (s)
    3. Explicit left/right representations
    4. Learned bilateral integration
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        latent_channels: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        self.activation = nn.ReLU()

        self.general_head = nn.Sequential(
            nn.Linear(
                hidden_channels,
                latent_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                latent_channels,
                latent_channels,
            ),
        )

        self.specific_head = nn.Sequential(
            nn.Linear(
                hidden_channels,
                latent_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                latent_channels,
                latent_channels,
            ),
        )

        # Left + right representation
        self.bilateral_fusion = nn.Sequential(
            nn.Linear(
                hidden_channels * 2,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
        )

        self.integration = nn.Sequential(
            nn.Linear(
                latent_channels * 2
                + hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ):
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        h = self.conv1(x, edge_index)
        h = self.activation(h)

        h = self.conv2(h, edge_index)
        h = self.activation(h)

        # Determine hemisphere using feature 2.
        left_mask = x[:, 2] < 0
        right_mask = x[:, 2] > 0

        # Graph-level hemisphere representations.
        left_h = global_mean_pool(
            h[left_mask],
            batch[left_mask],
        )

        right_h = global_mean_pool(
            h[right_mask],
            batch[right_mask],
        )

        bilateral = self.bilateral_fusion(
            torch.cat(
                [left_h, right_h],
                dim=-1,
            )
        )

        global_h = global_mean_pool(
            h,
            batch,
        )

        z_g = self.general_head(global_h)
        z_s = self.specific_head(global_h)

        combined = torch.cat(
            [
                z_g,
                z_s,
                bilateral,
            ],
            dim=-1,
        )

        integrated = self.integration(
            combined
        )

        logits = self.classifier(
            integrated
        )

        return logits, z_g, z_s, bilateral


if __name__ == "__main__":
    from src.data.bilateral_connectome import create_sample

    sample = create_sample()

    model = BilateralTwoFactorGNN(
        in_channels=8,
    )

    batch = torch.zeros(
        sample.x.size(0),
        dtype=torch.long,
    )

    logits, z_g, z_s, bilateral = model(
        sample.x,
        sample.edge_index,
        batch,
    )

    print("Input shape     :", sample.x.shape)
    print("General factor  :", z_g.shape)
    print("Specific factor :", z_s.shape)
    print("Bilateral state :", bilateral.shape)
    print("Output shape    :", logits.shape)
    print("Model test      : PASS")
