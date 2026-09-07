from dataclasses import dataclass
from typing import Callable
import math
import torch


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    difficulty: float
    input_dim: int
    output_dim: int
    generator: Callable[[int, int, int, torch.Generator], tuple[torch.Tensor, torch.Tensor]]


class CognitiveBattery:
    def __init__(self, input_dim: int = 16, seed: int = 20260907):
        self.input_dim = input_dim
        self.seed = seed
        self.tasks = self._build_tasks()

    def _generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _classification_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        mode: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, self.input_dim, generator=generator)
        noise = 0.15 + 0.85 * difficulty

        if mode == 0:
            score = x[:, 0] + 0.5 * x[:, 1]
        elif mode == 1:
            score = x[:, 0] * x[:, 1]
        elif mode == 2:
            score = x[:, 0] - x[:, 1] + 0.4 * x[:, 2]
        elif mode == 3:
            score = torch.sin(x[:, 0]) + 0.5 * x[:, 1]
        elif mode == 4:
            score = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]
        else:
            score = x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]

        score = score + noise * torch.randn(n, generator=generator)
        y = (score > 0).long()
        return x, y

    def _memory_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        position: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, 2, (n, self.input_dim), generator=generator).float()
        target_position = min(position, self.input_dim - 1)
        target = x[:, target_position].long()
        noise_probability = 0.05 + 0.30 * difficulty
        flips = torch.rand(n, generator=generator) < noise_probability
        target = torch.where(flips, 1 - target, target)
        return x, target

    def _control_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        rule: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, self.input_dim, generator=generator)
        if rule == 0:
            target = ((x[:, 0] > 0) != (x[:, 1] > 0)).long()
        else:
            target = ((x[:, 0] > 0) == (x[:, 1] > 0)).long()
        distractor = difficulty * torch.randn(n, self.input_dim, generator=generator)
        x = x + distractor
        return x, target

    def _reasoning_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        rule: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, 10, (n, self.input_dim), generator=generator).float()

        if rule == 0:
            score = x[:, 0] + x[:, 1] - x[:, 2]
        elif rule == 1:
            score = x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3]
        else:
            score = x[:, 0] + x[:, 1] + x[:, 2] - x[:, 3]

        threshold = 10.0 + 4.0 * difficulty
        target = (score > threshold).long()
        return x / 9.0, target

    def _decision_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        rule: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, self.input_dim, generator=generator)
        evidence = x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2]
        evidence = evidence / (1.0 + difficulty)
        evidence = evidence + difficulty * torch.randn(n, generator=generator)
        target = (evidence > 0).long()

        if rule:
            target = 1 - target

        return x, target

    def _generalization_task(
        self,
        n: int,
        difficulty: float,
        task_seed: int,
        rule: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, self.input_dim, generator=generator)

        if rule == 0:
            score = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]
        elif rule == 1:
            score = torch.sin(x[:, 0] + x[:, 1]) + x[:, 2]
        else:
            score = x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3]

        score = score + difficulty * torch.randn(n, generator=generator)
        target = (score > 0).long()
        return x, target

    def _build_tasks(self) -> list[TaskSpec]:
        return [
            TaskSpec("PER01", "perception", 0.20, self.input_dim, 2, lambda n, d, s, g: self._classification_task(n, d, s, 0, g)),
            TaskSpec("PER02", "perception", 0.40, self.input_dim, 2, lambda n, d, s, g: self._classification_task(n, d, s, 1, g)),
            TaskSpec("PER03", "perception", 0.70, self.input_dim, 2, lambda n, d, s, g: self._classification_task(n, d, s, 2, g)),
            TaskSpec("WM01", "working_memory", 0.20, self.input_dim, 2, lambda n, d, s, g: self._memory_task(n, d, s, 2, g)),
            TaskSpec("WM02", "working_memory", 0.50, self.input_dim, 2, lambda n, d, s, g: self._memory_task(n, d, s, 7, g)),
            TaskSpec("WM03", "working_memory", 0.80, self.input_dim, 2, lambda n, d, s, g: self._memory_task(n, d, s, 12, g)),
            TaskSpec("CTL01", "cognitive_control", 0.20, self.input_dim, 2, lambda n, d, s, g: self._control_task(n, d, s, 0, g)),
            TaskSpec("CTL02", "cognitive_control", 0.50, self.input_dim, 2, lambda n, d, s, g: self._control_task(n, d, s, 1, g)),
            TaskSpec("CTL03", "cognitive_control", 0.80, self.input_dim, 2, lambda n, d, s, g: self._control_task(n, d, s, 0, g)),
            TaskSpec("RSN01", "reasoning", 0.25, self.input_dim, 2, lambda n, d, s, g: self._reasoning_task(n, d, s, 0, g)),
            TaskSpec("RSN02", "reasoning", 0.55, self.input_dim, 2, lambda n, d, s, g: self._reasoning_task(n, d, s, 1, g)),
            TaskSpec("RSN03", "reasoning", 0.85, self.input_dim, 2, lambda n, d, s, g: self._reasoning_task(n, d, s, 2, g)),
            TaskSpec("DEC01", "decision", 0.20, self.input_dim, 2, lambda n, d, s, g: self._decision_task(n, d, s, 0, g)),
            TaskSpec("DEC02", "decision", 0.50, self.input_dim, 2, lambda n, d, s, g: self._decision_task(n, d, s, 1, g)),
            TaskSpec("DEC03", "decision", 0.80, self.input_dim, 2, lambda n, d, s, g: self._decision_task(n, d, s, 0, g)),
            TaskSpec("GEN01", "generalization", 0.25, self.input_dim, 2, lambda n, d, s, g: self._generalization_task(n, d, s, 0, g)),
            TaskSpec("GEN02", "generalization", 0.55, self.input_dim, 2, lambda n, d, s, g: self._generalization_task(n, d, s, 1, g)),
            TaskSpec("GEN03", "generalization", 0.85, self.input_dim, 2, lambda n, d, s, g: self._generalization_task(n, d, s, 2, g)),
        ]

    def generate(
        self,
        task_id: str,
        n: int = 256,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task = next(task for task in self.tasks if task.task_id == task_id)
        task_seed = self.seed if seed is None else seed
        generator = self._generator(task_seed)
        return task.generator(n, task.difficulty, task_seed, generator)

    def sample_all(
        self,
        n: int = 256,
        seed: int | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        base_seed = self.seed if seed is None else seed
        samples = {}

        for index, task in enumerate(self.tasks):
            samples[task.task_id] = self.generate(
                task.task_id,
                n=n,
                seed=base_seed + index,
            )

        return samples

    def summary(self) -> list[dict]:
        return [
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "input_dim": task.input_dim,
                "output_dim": task.output_dim,
            }
            for task in self.tasks
        ]


if __name__ == "__main__":
    battery = CognitiveBattery()
    samples = battery.sample_all(n=64)

    print(f"Tasks: {len(battery.tasks)}")
    print(f"Domains: {len(set(task.domain for task in battery.tasks))}")

    for task in battery.tasks:
        x, y = samples[task.task_id]
        print(
            f"{task.task_id} | {task.domain} | "
            f"difficulty={task.difficulty:.2f} | "
            f"x={tuple(x.shape)} | y={tuple(y.shape)}"
        )
