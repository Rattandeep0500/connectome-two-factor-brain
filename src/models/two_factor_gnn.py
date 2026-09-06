import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class TwoFactorGNN(nn.Module):
    """
    Spearman-inspired two-factor connectome GNN.

    z_g: task-general representation
    z_s: task-specific representation
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        latent_channels: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()

        # Shared graph encoder
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        self.activation = nn.ReLU()

        # General cognitive factor pathway
        self.general_head = nn.Sequential(
            nn.Linear(hidden_channels, latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels),
        )

        # Task-specific factor pathway
        self.specific_head = nn.Sequential(
            nn.Linear(hidden_channels, latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels),
        )

        # Learned integration
        self.integration = nn.Sequential(
            nn.Linear(latent_channels * 2, hidden_channels),
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

        # Graph representation
        x = self.conv1(x, edge_index)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.activation(x)

        x = global_mean_pool(x, batch)

        # Two-factor decomposition
        z_g = self.general_head(x)
        z_s = self.specific_head(x)

        # Integration
        z = torch.cat(
            [z_g, z_s],
            dim=-1,
        )

        z = self.integration(z)

        logits = self.classifier(z)

        return logits, z_g, z_s


if __name__ == "__main__":
    num_nodes = 20
    num_features = 8
    num_edges = 60

    x = torch.randn(
        num_nodes,
        num_features,
    )

    edge_index = torch.randint(
        0,
        num_nodes,
        (2, num_edges),
    )

    batch = torch.zeros(
        num_nodes,
        dtype=torch.long,
    )

    model = TwoFactorGNN(
        in_channels=num_features,
    )

    logits, z_g, z_s = model(
        x,
        edge_index,
        batch,
    )

    print("Input shape      :", x.shape)
    print("General factor   :", z_g.shape)
    print("Specific factor  :", z_s.shape)
    print("Output shape     :", logits.shape)
    print("Model test       : PASS")
