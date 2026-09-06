import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class ConnectomeGNN(nn.Module):
    """Baseline graph-level GNN for connectome classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.activation(x)

        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        x = global_mean_pool(x, batch)

        return self.classifier(x)


if __name__ == "__main__":
    num_nodes = 20
    num_features = 8
    num_edges = 60

    x = torch.randn(num_nodes, num_features)

    edge_index = torch.randint(
        0,
        num_nodes,
        (2, num_edges),
    )

    batch = torch.zeros(
        num_nodes,
        dtype=torch.long,
    )

    model = ConnectomeGNN(
        in_channels=num_features,
        hidden_channels=64,
        num_classes=2,
    )

    output = model(
        x,
        edge_index,
        batch,
    )

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print("Model test  : PASS")
