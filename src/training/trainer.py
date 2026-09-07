from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TaskMetrics:
    task_id: str
    domain: str
    loss: float
    accuracy: float
    confidence: float
    samples: int


class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        task_ids: list[str],
        device: str = "cpu",
        learning_rate: float = 0.001,
    ):
        self.model = model.to(device)
        self.task_ids = task_ids
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        self.criterion = nn.CrossEntropyLoss()

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, float]:
        self.model.train()

        x = x.to(self.device)
        y = y.to(self.device)
        task_index = task_index.to(self.device)

        self.optimizer.zero_grad()

        output = self.model(x, task_index)

        loss = self.criterion(
            output["logits"],
            y,
        )

        loss.backward()
        self.optimizer.step()

        predictions = output["logits"].argmax(dim=-1)
        accuracy = (
            predictions == y
        ).float().mean()

        confidence = output["confidence"].mean()

        return {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "confidence": float(confidence.detach()),
        }

    @torch.no_grad()
    def evaluate_task(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_index: int,
        task_id: str,
        domain: str,
    ) -> TaskMetrics:
        self.model.eval()

        x = x.to(self.device)
        y = y.to(self.device)

        task_tensor = torch.full(
            (x.shape[0],),
            task_index,
            dtype=torch.long,
            device=self.device,
        )

        output = self.model(
            x,
            task_tensor,
        )

        loss = self.criterion(
            output["logits"],
            y,
        )

        predictions = output["logits"].argmax(dim=-1)

        accuracy = (
            predictions == y
        ).float().mean()

        confidence = output["confidence"].mean()

        return TaskMetrics(
            task_id=task_id,
            domain=domain,
            loss=float(loss),
            accuracy=float(accuracy),
            confidence=float(confidence),
            samples=int(x.shape[0]),
        )

    def evaluate_battery(
        self,
        environment,
        n: int = 256,
        split: str = "evaluation",
        seed: int = 20260907,
    ) -> list[TaskMetrics]:
        results = []

        for task_index, task in enumerate(environment.tasks):
            batch = environment.generate(
                task.task_id,
                n=n,
                split=split,
                seed=seed + task_index * 1009,
            )

            metrics = self.evaluate_task(
                batch.x,
                batch.y,
                task_index,
                task.task_id,
                task.domain,
            )

            results.append(metrics)

        return results

    def aggregate_metrics(
        self,
        metrics: list[TaskMetrics],
    ) -> dict[str, float]:
        losses = torch.tensor(
            [metric.loss for metric in metrics],
            dtype=torch.float32,
        )

        accuracies = torch.tensor(
            [metric.accuracy for metric in metrics],
            dtype=torch.float32,
        )

        confidences = torch.tensor(
            [metric.confidence for metric in metrics],
            dtype=torch.float32,
        )

        return {
            "mean_loss": float(losses.mean()),
            "mean_accuracy": float(accuracies.mean()),
            "mean_confidence": float(confidences.mean()),
        }


if __name__ == "__main__":
    from src.experiments.cognitive_environment import CognitiveEnvironment
    from src.models.generic_recurrent import GenericRecurrentBrain

    environment = CognitiveEnvironment()

    model = GenericRecurrentBrain()

    trainer = MultiTaskTrainer(
        model=model,
        task_ids=[
            task.task_id
            for task in environment.tasks
        ],
    )

    batch = environment.generate(
        "RSN02",
        n=64,
        split="training",
        seed=123,
    )

    task_index = torch.full(
        (64,),
        10,
        dtype=torch.long,
    )

    metrics = trainer.train_step(
        batch.x,
        batch.y,
        task_index,
    )

    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Confidence: {metrics['confidence']:.6f}")

    results = trainer.evaluate_battery(
        environment,
        n=32,
    )

    aggregate = trainer.aggregate_metrics(
        results
    )

    print(f"Evaluation loss: {aggregate['mean_loss']:.6f}")
    print(f"Evaluation accuracy: {aggregate['mean_accuracy']:.6f}")
    print(f"Evaluation confidence: {aggregate['mean_confidence']:.6f}")
    print("Training harness test: PASS")
