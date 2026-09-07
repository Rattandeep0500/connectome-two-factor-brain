from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TrialBatch:
    x: torch.Tensor
    y: torch.Tensor
    task_id: str
    domain: str
    difficulty: float
    split: str
    seed: int


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    domain: str
    difficulty: float
    demand_profile: tuple[float, ...]
    task_family: str


class CognitiveEnvironment:
    def __init__(
        self,
        input_dim: int = 16,
        seed: int = 20260907,
    ):
        self.input_dim = input_dim
        self.seed = seed
        self.demands = (
            "perceptual_discrimination",
            "maintenance",
            "interference_control",
            "relational_reasoning",
            "evidence_integration",
            "abstraction",
        )
        self.tasks = self._build_tasks()

    def _build_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition("PER01", "perception", 0.20, (0.90, 0.15, 0.10, 0.10, 0.05, 0.10), "linear"),
            TaskDefinition("PER02", "perception", 0.45, (0.85, 0.20, 0.10, 0.10, 0.05, 0.20), "nonlinear"),
            TaskDefinition("PER03", "perception", 0.75, (0.80, 0.25, 0.15, 0.15, 0.10, 0.25), "relational"),
            TaskDefinition("WM01", "working_memory", 0.20, (0.10, 0.90, 0.25, 0.10, 0.05, 0.10), "maintenance"),
            TaskDefinition("WM02", "working_memory", 0.50, (0.10, 0.85, 0.45, 0.15, 0.10, 0.15), "maintenance"),
            TaskDefinition("WM03", "working_memory", 0.80, (0.15, 0.80, 0.65, 0.20, 0.10, 0.20), "interference"),
            TaskDefinition("CTL01", "cognitive_control", 0.20, (0.10, 0.25, 0.90, 0.15, 0.10, 0.10), "inhibition"),
            TaskDefinition("CTL02", "cognitive_control", 0.50, (0.10, 0.30, 0.85, 0.25, 0.15, 0.15), "rule_switch"),
            TaskDefinition("CTL03", "cognitive_control", 0.80, (0.15, 0.35, 0.80, 0.35, 0.20, 0.20), "conflict"),
            TaskDefinition("RSN01", "reasoning", 0.25, (0.15, 0.30, 0.20, 0.90, 0.15, 0.30), "additive"),
            TaskDefinition("RSN02", "reasoning", 0.55, (0.15, 0.35, 0.25, 0.85, 0.20, 0.50), "relational"),
            TaskDefinition("RSN03", "reasoning", 0.85, (0.20, 0.40, 0.30, 0.80, 0.25, 0.75), "compositional"),
            TaskDefinition("DEC01", "decision", 0.20, (0.10, 0.20, 0.20, 0.20, 0.90, 0.15), "evidence"),
            TaskDefinition("DEC02", "decision", 0.50, (0.15, 0.25, 0.30, 0.25, 0.85, 0.25), "evidence"),
            TaskDefinition("DEC03", "decision", 0.80, (0.20, 0.30, 0.35, 0.30, 0.80, 0.40), "conflict_evidence"),
            TaskDefinition("GEN01", "generalization", 0.25, (0.20, 0.25, 0.15, 0.35, 0.20, 0.90), "abstraction"),
            TaskDefinition("GEN02", "generalization", 0.55, (0.25, 0.30, 0.20, 0.45, 0.25, 0.85), "transfer"),
            TaskDefinition("GEN03", "generalization", 0.85, (0.30, 0.35, 0.30, 0.55, 0.30, 0.80), "compositional_transfer"),
        ]

    def _generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _binary_target(
        self,
        score: torch.Tensor,
        generator: torch.Generator,
        difficulty: float,
    ) -> torch.Tensor:
        noise_scale = 0.10 + 0.35 * difficulty
        noisy_score = score + noise_scale * torch.randn(
            score.shape,
            generator=generator,
        )
        return (noisy_score > 0).long()

    def _generate_trial(
        self,
        task: TaskDefinition,
        n: int,
        seed: int,
        split: str,
    ) -> TrialBatch:
        generator = self._generator(seed)
        x = torch.randn(
            n,
            self.input_dim,
            generator=generator,
        )

        d = task.demand_profile
        family = task.task_family

        if family == "linear":
            score = x[:, 0] * d[0] + x[:, 1] * d[1] + x[:, 2] * d[2]

        elif family == "nonlinear":
            score = (
                x[:, 0] * x[:, 1] * d[0]
                + x[:, 2] * d[1]
                - x[:, 3] * d[2]
            )

        elif family == "relational":
            score = (
                (x[:, 0] - x[:, 1]) * d[0]
                + (x[:, 2] * x[:, 3]) * d[3]
                + x[:, 4] * d[5]
            )

        elif family == "maintenance":
            score = (
                x[:, 2] * d[1]
                + x[:, 7] * d[0]
                - x[:, 9] * d[2]
            )

        elif family == "interference":
            score = (
                x[:, 2] * d[1]
                - x[:, 7] * d[2]
                + x[:, 9] * d[2]
                + x[:, 11] * d[0]
            )

        elif family == "inhibition":
            score = (
                x[:, 0] * d[0]
                - x[:, 1] * d[2]
                + x[:, 2] * d[1]
            )

        elif family == "rule_switch":
            rule = torch.where(
                x[:, 4] > 0,
                x[:, 0] - x[:, 1],
                x[:, 0] + x[:, 1],
            )
            score = rule * d[2] + x[:, 2] * d[1]

        elif family == "conflict":
            score = (
                x[:, 0] * d[0]
                - x[:, 1] * d[2]
                + x[:, 2] * d[3]
                - x[:, 3] * d[2]
            )

        elif family == "additive":
            score = (
                x[:, 0] * d[3]
                + x[:, 1] * d[3]
                - x[:, 2] * d[1]
            )

        elif family == "compositional":
            score = (
                x[:, 0] * x[:, 1] * d[3]
                - x[:, 2] * x[:, 3] * d[5]
                + x[:, 4] * d[1]
            )

        elif family == "evidence":
            score = (
                x[:, 0] * d[4]
                + 0.7 * x[:, 1] * d[4]
                - 0.4 * x[:, 2] * d[2]
            )

        elif family == "conflict_evidence":
            score = (
                x[:, 0] * d[4]
                - x[:, 1] * d[2]
                + x[:, 2] * d[3]
                - x[:, 3] * d[0]
            )

        elif family == "abstraction":
            score = (
                torch.sin(x[:, 0] + x[:, 1]) * d[5]
                + x[:, 2] * d[3]
                - x[:, 3] * d[1]
            )

        elif family == "transfer":
            score = (
                x[:, 0] * x[:, 1] * d[5]
                + torch.sin(x[:, 2]) * d[3]
                - x[:, 3] * d[1]
            )

        elif family == "compositional_transfer":
            score = (
                x[:, 0] * x[:, 1] * d[5]
                - x[:, 2] * x[:, 3] * d[3]
                + torch.sin(x[:, 4] + x[:, 5]) * d[5]
            )

        else:
            raise ValueError(f"Unknown task family: {family}")

        y = self._binary_target(
            score,
            generator,
            task.difficulty,
        )

        return TrialBatch(
            x=x,
            y=y,
            task_id=task.task_id,
            domain=task.domain,
            difficulty=task.difficulty,
            split=split,
            seed=seed,
        )

    def task(self, task_id: str) -> TaskDefinition:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def generate(
        self,
        task_id: str,
        n: int = 256,
        split: str = "training",
        seed: int | None = None,
    ) -> TrialBatch:
        if split not in {"training", "evaluation"}:
            raise ValueError("split must be 'training' or 'evaluation'")

        task = self.task(task_id)
        base_seed = self.seed if seed is None else seed

        if split == "evaluation":
            base_seed += 1000003

        return self._generate_trial(
            task,
            n,
            base_seed,
            split,
        )

    def generate_all(
        self,
        n: int = 256,
        split: str = "training",
        seed: int | None = None,
    ) -> dict[str, TrialBatch]:
        base_seed = self.seed if seed is None else seed

        return {
            task.task_id: self.generate(
                task.task_id,
                n=n,
                split=split,
                seed=base_seed + index * 1009,
            )
            for index, task in enumerate(self.tasks)
        }

    def task_matrix(self) -> list[dict]:
        return [
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "task_family": task.task_family,
                "demand_profile": task.demand_profile,
            }
            for task in self.tasks
        ]

    def summary(self) -> dict:
        return {
            "tasks": len(self.tasks),
            "domains": len(set(task.domain for task in self.tasks)),
            "input_dim": self.input_dim,
            "demands": self.demands,
        }


if __name__ == "__main__":
    environment = CognitiveEnvironment()

    print(f"Tasks: {len(environment.tasks)}")
    print(f"Domains: {len(set(task.domain for task in environment.tasks))}")
    print(f"Input dimension: {environment.input_dim}")
    print(f"Demand dimensions: {len(environment.demands)}")

    training = environment.generate_all(
        n=64,
        split="training",
        seed=20260907,
    )

    evaluation = environment.generate_all(
        n=64,
        split="evaluation",
        seed=20260907,
    )

    for task in environment.tasks:
        train_batch = training[task.task_id]
        eval_batch = evaluation[task.task_id]

        print(
            f"{task.task_id} | {task.domain} | "
            f"train={tuple(train_batch.x.shape)} | "
            f"eval={tuple(eval_batch.x.shape)} | "
            f"demands={task.demand_profile}"
        )

    print(
        "Training/evaluation inputs identical:",
        bool(torch.equal(
            training["PER01"].x,
            evaluation["PER01"].x,
        )),
    )

