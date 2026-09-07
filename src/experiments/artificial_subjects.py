from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SubjectProfile:
    subject_id: int
    seed: int
    training_steps: int
    learning_rate: float
    experience_scale: float
    domain_exposure: tuple[float, ...]


class ArtificialSubjectPopulation:
    def __init__(
        self,
        n_subjects: int = 128,
        n_domains: int = 6,
        seed: int = 20260907,
    ):
        self.n_subjects = n_subjects
        self.n_domains = n_domains
        self.seed = seed
        self.subjects = self._build_population()

    def _build_population(self) -> list[SubjectProfile]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        subjects = []

        for subject_id in range(self.n_subjects):
            subject_seed = self.seed + subject_id * 1009

            training_steps = int(
                torch.randint(
                    800,
                    2401,
                    (1,),
                    generator=generator,
                ).item()
            )

            learning_rate_log = -3.4 + 1.0 * torch.rand(
                1,
                generator=generator,
            ).item()

            learning_rate = float(10 ** learning_rate_log)

            experience_scale = float(
                0.75
                + 0.50
                * torch.rand(
                    1,
                    generator=generator,
                ).item()
            )

            domain_exposure = tuple(
                float(
                    0.75
                    + 0.50
                    * torch.rand(
                        1,
                        generator=generator,
                    ).item()
                )
                for _ in range(self.n_domains)
            )

            subjects.append(
                SubjectProfile(
                    subject_id=subject_id,
                    seed=subject_seed,
                    training_steps=training_steps,
                    learning_rate=learning_rate,
                    experience_scale=experience_scale,
                    domain_exposure=domain_exposure,
                )
            )

        return subjects

    def get(self, subject_id: int) -> SubjectProfile:
        return self.subjects[subject_id]

    def summary(self) -> dict:
        training_steps = torch.tensor(
            [subject.training_steps for subject in self.subjects],
            dtype=torch.float32,
        )

        learning_rates = torch.tensor(
            [subject.learning_rate for subject in self.subjects],
            dtype=torch.float32,
        )

        experience = torch.tensor(
            [subject.experience_scale for subject in self.subjects],
            dtype=torch.float32,
        )

        return {
            "n_subjects": self.n_subjects,
            "n_domains": self.n_domains,
            "training_steps_mean": float(training_steps.mean()),
            "training_steps_std": float(training_steps.std()),
            "learning_rate_mean": float(learning_rates.mean()),
            "learning_rate_std": float(learning_rates.std()),
            "experience_mean": float(experience.mean()),
            "experience_std": float(experience.std()),
        }


if __name__ == "__main__":
    population = ArtificialSubjectPopulation()

    print(f"Subjects: {population.n_subjects}")
    print(f"Domains: {population.n_domains}")

    summary = population.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    subject = population.get(0)

    print(f"Subject ID: {subject.subject_id}")
    print(f"Seed: {subject.seed}")
    print(f"Training steps: {subject.training_steps}")
    print(f"Learning rate: {subject.learning_rate:.6f}")
    print(f"Experience scale: {subject.experience_scale:.4f}")
    print(f"Domain exposure: {subject.domain_exposure}")
