import torch
import torch.nn as nn


class NeuralPopulation(nn.Module):
    def __init__(
        self,
        state_size: int = 32,
        biological_neurons: int = 1_000_000,
    ):
        super().__init__()

        self.state_size = state_size
        self.biological_neurons = biological_neurons

        self.state = nn.Parameter(
            torch.zeros(state_size),
            requires_grad=False,
        )

        self.update = nn.Linear(
            state_size,
            state_size,
        )

        self.activation = nn.Tanh()

    def reset(self):
        self.state.zero_()

    def step(
        self,
        input_signal: torch.Tensor,
    ) -> torch.Tensor:
        new_state = self.activation(
            self.update(self.state)
            + input_signal
        )

        self.state.copy_(new_state.detach())

        return self.state


if __name__ == "__main__":
    population = NeuralPopulation(
        state_size=32,
        biological_neurons=1_000_000,
    )

    print(
        "Biological-equivalent neurons:",
        population.biological_neurons,
    )

    print(
        "Initial state:",
        population.state.shape,
    )

    signal = torch.randn(32)

    state = population.step(signal)

    print(
        "Updated state:",
        state.shape,
    )

    population.reset()

    print(
        "Reset state:",
        population.state.shape,
    )

    print("Population test: PASS")
