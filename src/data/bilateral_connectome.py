import torch
from torch_geometric.data import Data


def create_sample(
    num_nodes_per_hemisphere: int = 20,
    node_features: int = 8,
):
    """
    Synthetic bilateral connectome.

    Features:
        feature 0 = general cognitive signal (g)
        feature 1 = task-specific signal (s)
        feature 2 = bilateral interaction signal

    Nodes 0..N-1:
        left hemisphere

    Nodes N..2N-1:
        right hemisphere
    """

    n = num_nodes_per_hemisphere
    total_nodes = 2 * n

    x = torch.randn(total_nodes, node_features)

    # Hemisphere indicator.
    # Negative = left, positive = right.
    x[:n, 2] = -1.0
    x[n:, 2] = 1.0

    # Shared general-factor signal.
    g_left = torch.randn(1)
    g_right = g_left + 0.15 * torch.randn(1)

    x[:n, 0] += g_left
    x[n:, 0] += g_right

    # Task-specific signals.
    s_left = torch.randn(1)
    s_right = torch.randn(1)

    x[:n, 1] += s_left
    x[n:, 1] += s_right

    # Within-hemisphere connectivity.
    edges = []

    for start, end in [(0, n), (n, 2 * n)]:
        for i in range(start, end):
            for j in range(i + 1, end):
                if torch.rand(()) < 0.15:
                    edges.append((i, j))
                    edges.append((j, i))

    # Cross-hemisphere connections.
    for i in range(n):
        j = n + i

        if torch.rand(()) < 0.35:
            edges.append((i, j))
            edges.append((j, i))

    # Guarantee every hemisphere has useful cross-links.
    for i in range(0, n, 5):
        j = n + i
        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()

    # Aggregate signals.
    general_signal = g_left.squeeze()

    task_signal = (
        s_left.squeeze() +
        s_right.squeeze()
    )

    bilateral_signal = (
        s_left.squeeze() *
        s_right.squeeze()
    )

    # Decision rule:
    # general ability + task-specific contribution
    # + bilateral integration.
    score = (
        0.8 * general_signal
        + 0.5 * task_signal
        + 0.9 * bilateral_signal
    )

    y = torch.tensor(
        [1 if score > 0 else 0],
        dtype=torch.long,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )


def create_dataset(num_samples: int = 1000):
    return [
        create_sample()
        for _ in range(num_samples)
    ]


if __name__ == "__main__":
    dataset = create_dataset(5)

    sample = dataset[0]

    print("Samples:", len(dataset))
    print("Nodes:", sample.x.shape)
    print("Edges:", sample.edge_index.shape)
    print("Label:", sample.y.item())
    print("Bilateral dataset test: PASS")
