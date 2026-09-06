import torch
from torch_geometric.data import Data


def create_sample(
    num_nodes: int = 40,
    node_features: int = 8,
):
    x = torch.randn(num_nodes, node_features)

    # Random undirected graph
    edges = torch.randint(
        0,
        num_nodes,
        (2, num_nodes * 5),
    )

    edge_index = torch.cat(
        [edges, edges.flip(0)],
        dim=1,
    )

    # Synthetic decision signal:
    # stronger aggregate node signal -> class 1
    score = x[:, 0].mean()

    y = torch.tensor(
        [1 if score > 0 else 0],
        dtype=torch.long,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )


def create_dataset(
    num_samples: int = 1000,
):
    return [
        create_sample()
        for _ in range(num_samples)
    ]


if __name__ == "__main__":
    dataset = create_dataset(5)

    print("Samples:", len(dataset))
    print("Nodes:", dataset[0].x.shape)
    print("Edges:", dataset[0].edge_index.shape)
    print("Label:", dataset[0].y.item())
    print("Dataset test: PASS")
