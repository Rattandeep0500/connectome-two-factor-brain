import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.data.synthetic_connectome import create_dataset
from src.models.two_factor_gnn import TwoFactorGNN


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == batch.y
            ).sum().item()

            total += batch.y.size(0)

    return correct / total


def main():
    torch.manual_seed(42)

    dataset = create_dataset(1000)

    train_set = dataset[:800]
    test_set = dataset[800:]

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
    )

    model = TwoFactorGNN(
        in_channels=8,
        hidden_channels=64,
        latent_channels=32,
        num_classes=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_accuracy = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            logits, z_g, z_s = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            classification_loss = F.cross_entropy(
                logits,
                batch.y,
            )

            # Mild decorrelation between the two latent pathways.
            g_norm = F.normalize(z_g, dim=-1)
            s_norm = F.normalize(z_s, dim=-1)

            factor_overlap = (
                g_norm * s_norm
            ).sum(dim=-1).abs().mean()

            loss = (
                classification_loss
                + 0.01 * factor_overlap
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        accuracy = evaluate(
            model,
            test_loader,
        )

        average_loss = (
            total_loss / len(train_loader)
        )

        history.append({
            "epoch": epoch,
            "loss": average_loss,
            "test_accuracy": accuracy,
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                RESULTS_DIR / "two_factor_gnn_best.pt",
            )

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Loss {average_loss:.4f} | "
                f"Test Accuracy {accuracy:.4f}"
            )

    results = {
        "model": "TwoFactorGNN",
        "dataset": "synthetic_connectome",
        "seed": 42,
        "best_test_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "baseline_accuracy": 0.9550,
        "history": history,
    }

    with open(
        RESULTS_DIR / "two_factor_gnn_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    print()
    print(
        f"Two-Factor Best Test Accuracy: "
        f"{best_accuracy:.4f}"
    )
    print(f"Best Epoch: {best_epoch}")
    print(
        "Checkpoint saved: "
        "results/two_factor_gnn_best.pt"
    )
    print(
        "Results saved: "
        "results/two_factor_gnn_results.json"
    )


if __name__ == "__main__":
    main()
