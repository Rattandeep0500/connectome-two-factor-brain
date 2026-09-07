import json
import random
from pathlib import Path

import numpy as np
import torch

from src.experiments.cognitive_environment import CognitiveEnvironment
from src.models.generic_recurrent import GenericRecurrentBrain
from src.training.trainer import MultiTaskTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    seed: int = 20260907,
    steps: int = 2000,
    batch_size: int = 64,
):
    set_seed(seed)

    environment = CognitiveEnvironment(
        input_dim=16,
        seed=seed,
    )

    task_ids = [
        task.task_id
        for task in environment.tasks
    ]

    model = GenericRecurrentBrain(
        input_dim=16,
        hidden_dim=64,
        n_tasks=len(task_ids),
        task_embedding_dim=16,
        rollout_steps=8,
        output_dim=2,
    )

    trainer = MultiTaskTrainer(
        model=model,
        task_ids=task_ids,
        learning_rate=0.001,
    )

    history = []

    for step in range(1, steps + 1):
        task_index = (step - 1) % len(task_ids)
        task_id = task_ids[task_index]

        batch = environment.generate(
            task_id,
            n=batch_size,
            split="training",
            seed=seed + step * 7919,
        )

        task_tensor = torch.full(
            (batch_size,),
            task_index,
            dtype=torch.long,
        )

        metrics = trainer.train_step(
            batch.x,
            batch.y,
            task_tensor,
        )

        history.append(
            {
                "step": step,
                "task_id": task_id,
                **metrics,
            }
        )

        if step == 1 or step % 100 == 0:
            print(
                f"Step {step:04d} | "
                f"Task {task_id} | "
                f"Loss {metrics['loss']:.6f} | "
                f"Accuracy {metrics['accuracy']:.4f}"
            )

    evaluation = trainer.evaluate_battery(
        environment,
        n=512,
        split="evaluation",
        seed=seed + 500000,
    )

    aggregate = trainer.aggregate_metrics(
        evaluation
    )

    task_results = {}

    for metric in evaluation:
        task_results[metric.task_id] = {
            "domain": metric.domain,
            "loss": metric.loss,
            "accuracy": metric.accuracy,
            "confidence": metric.confidence,
            "samples": metric.samples,
        }

    domain_results = {}

    for metric in evaluation:
        domain_results.setdefault(
            metric.domain,
            [],
        ).append(metric.accuracy)

    domain_results = {
        domain: {
            "accuracy": float(
                np.mean(values)
            )
        }
        for domain, values in domain_results.items()
    }

    results = {
        "experiment": "M1_generic_recurrent",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "parameter_count": model.parameter_count(),
        "hidden_dim": model.hidden_dim,
        "rollout_steps": model.rollout_steps,
        "n_tasks": len(task_ids),
        "aggregate": aggregate,
        "tasks": task_results,
        "domains": domain_results,
        "history": history,
    }

    results_dir = Path("results")
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        results_dir
        / f"m1_generic_recurrent_seed_{seed}.pt"
    )

    results_path = (
        results_dir
        / f"m1_generic_recurrent_seed_{seed}.json"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "parameter_count": model.parameter_count(),
        },
        checkpoint_path,
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("M1 COMPLETE")
    print(f"Parameters: {model.parameter_count()}")
    print(f"Evaluation loss: {aggregate['mean_loss']:.6f}")
    print(f"Evaluation accuracy: {aggregate['mean_accuracy']:.6f}")
    print(f"Evaluation confidence: {aggregate['mean_confidence']:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {results_path}")

    return results


if __name__ == "__main__":
    train()
