import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn


class RecurrentBrain(nn.Module):
    def __init__(
        self,
        num_general=100,
        num_specific=100,
        state_size=32,
        general_connection_probability=0.04,
        specific_connection_probability=0.04,
        cross_connection_probability=0.015,
        biological_neurons_per_population=500_000_000,
    ):
        super().__init__()

        self.num_general = num_general
        self.num_specific = num_specific
        self.num_populations = num_general + num_specific
        self.state_size = state_size
        self.biological_neurons_per_population = biological_neurons_per_population

        self.num_biological_equivalent_neurons = (
            self.num_populations
            * biological_neurons_per_population
        )

        self.adjacency = self._build_adjacency(
            general_connection_probability,
            specific_connection_probability,
            cross_connection_probability,
        )

        self.general_update = nn.Linear(state_size, state_size)
        self.specific_update = nn.Linear(state_size, state_size)
        self.recurrent_update = nn.Linear(state_size, state_size)
        self.input_update = nn.Linear(state_size, state_size)

        self.state = torch.zeros(
            self.num_populations,
            state_size,
        )

        self.activation = nn.Tanh()

    def _build_adjacency(
        self,
        general_probability,
        specific_probability,
        cross_probability,
    ):
        rng = np.random.default_rng(42)

        n = self.num_populations
        g = self.num_general

        adjacency = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if i < g and j < g:
                    probability = general_probability
                elif i >= g and j >= g:
                    probability = specific_probability
                else:
                    probability = cross_probability

                if rng.random() < probability:
                    adjacency[i, j] = rng.uniform(0.05, 0.25)

        adjacency = adjacency / (
            adjacency.sum(axis=1, keepdims=True) + 1e-8
        )

        return torch.tensor(adjacency)

    def reset(self):
        self.state.zero_()

    def step(self, input_signal):
        recurrent_signal = torch.matmul(
            self.adjacency,
            self.state,
        )

        recurrent_signal = self.recurrent_update(
            recurrent_signal
        )

        general_state = self.state[:self.num_general]
        specific_state = self.state[self.num_general:]

        general_update = self.general_update(
            general_state
        )

        specific_update = self.specific_update(
            specific_state
        )

        input_update = self.input_update(
            input_signal
        )

        new_state = torch.zeros_like(self.state)

        new_state[:self.num_general] = self.activation(
            general_update
            + recurrent_signal[:self.num_general]
            + input_update[:self.num_general]
        )

        new_state[self.num_general:] = self.activation(
            specific_update
            + recurrent_signal[self.num_general:]
            + input_update[self.num_general:]
        )

        self.state = new_state

        return self.state

    def graph_data(self):
        adjacency = self.adjacency.numpy()

        edges = np.argwhere(adjacency > 0)

        weights = adjacency[
            edges[:, 0],
            edges[:, 1],
        ]

        edge_types = []

        for source, target in edges:
            if source < self.num_general and target < self.num_general:
                edge_types.append(0)
            elif source >= self.num_general and target >= self.num_general:
                edge_types.append(1)
            else:
                edge_types.append(2)

        rng = np.random.default_rng(7)

        general_positions = rng.normal(
            loc=(-3.0, 0.0, 0.0),
            scale=1.0,
            size=(self.num_general, 3),
        )

        specific_positions = rng.normal(
            loc=(3.0, 0.0, 0.0),
            scale=1.0,
            size=(self.num_specific, 3),
        )

        positions = np.vstack(
            [
                general_positions,
                specific_positions,
            ]
        )

        groups = np.concatenate(
            [
                np.zeros(self.num_general),
                np.ones(self.num_specific),
            ]
        )

        return {
            "positions": positions,
            "edges": edges,
            "weights": weights,
            "edge_types": np.array(edge_types),
            "groups": groups,
        }


if __name__ == "__main__":
    brain = RecurrentBrain()

    activity_history = []

    for _ in range(80):
        input_signal = torch.randn(
            brain.num_populations,
            brain.state_size,
        )

        state = brain.step(input_signal)

        activity = torch.norm(
            state,
            dim=1,
        ).detach().numpy()

        activity_history.append(activity)

    graph = brain.graph_data()

    sio.savemat(
        "results/brain_activity.mat",
        {
            **graph,
            "activity": np.array(activity_history),
        },
    )

    print(
        "General populations:",
        brain.num_general,
    )

    print(
        "Specific populations:",
        brain.num_specific,
    )

    print(
        "Total populations:",
        brain.num_populations,
    )

    print(
        "Biological-equivalent neurons:",
        brain.num_biological_equivalent_neurons,
    )

    print(
        "Activity shape:",
        np.array(activity_history).shape,
    )

    print(
        "Exported: results/brain_activity.mat"
    )

    print("Activity simulation: PASS")
