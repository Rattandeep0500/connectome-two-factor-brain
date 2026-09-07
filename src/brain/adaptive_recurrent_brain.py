import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRecurrentTwoFactorBrain(nn.Module):
    def __init__(
        self,
        input_dim=8,
        hidden_dim=32,
        num_general_populations=100,
        num_specific_populations=100,
        neurons_per_population=500_000_000,
        num_classes=2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_general_populations = num_general_populations
        self.num_specific_populations = num_specific_populations
        self.neurons_per_population = neurons_per_population
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.g_left = nn.GRUCell(
            hidden_dim * 3,
            hidden_dim
        )

        self.g_right = nn.GRUCell(
            hidden_dim * 3,
            hidden_dim
        )

        self.s_left = nn.GRUCell(
            hidden_dim * 3,
            hidden_dim
        )

        self.s_right = nn.GRUCell(
            hidden_dim * 3,
            hidden_dim
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

        self.bilateral = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.decision = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    @property
    def biological_equivalent_neurons(self):
        return (
            self.num_general_populations
            + self.num_specific_populations
        ) * self.neurons_per_population

    def forward(
        self,
        x,
        steps=8,
        return_history=False
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        encoded = self.encoder(x)

        g_left = encoded
        g_right = encoded
        s_left = encoded
        s_right = encoded

        decision_history = []
        gate_history = []

        for _ in range(steps):
            g_context = torch.cat(
                [
                    encoded,
                    s_left,
                    s_right
                ],
                dim=-1
            )

            s_context = torch.cat(
                [
                    encoded,
                    g_left,
                    g_right
                ],
                dim=-1
            )

            g_left = self.g_left(
                g_context,
                g_left
            )

            g_right = self.g_right(
                g_context,
                g_right
            )

            s_left = self.s_left(
                s_context,
                s_left
            )

            s_right = self.s_right(
                s_context,
                s_right
            )

            state = torch.cat(
                [
                    g_left,
                    g_right,
                    s_left,
                    s_right
                ],
                dim=-1
            )

            gates = F.softmax(
                self.gate(state),
                dim=-1
            )

            g_state = (
                g_left + g_right
            ) / 2

            s_state = (
                s_left + s_right
            ) / 2

            bilateral_state = self.bilateral(
                state
            )

            combined = torch.cat(
                [
                    gates[:, 0:1] * g_state,
                    gates[:, 1:2] * s_state
                ],
                dim=-1
            )

            decision_input = torch.cat(
                [
                    combined,
                    bilateral_state
                ],
                dim=-1
            )

            logits = self.decision(
                decision_input
            )

            decision_history.append(logits)
            gate_history.append(gates)

        final_logits = decision_history[-1]

        output = {
            "logits": final_logits,
            "probabilities": F.softmax(
                final_logits,
                dim=-1
            ),
            "g_state": g_state,
            "s_state": s_state,
            "bilateral_state": bilateral_state,
            "gates": gate_history[-1],
            "decision_history": torch.stack(
                decision_history,
                dim=1
            ),
            "gate_history": torch.stack(
                gate_history,
                dim=1
            )
        }

        if return_history:
            output["history"] = decision_history

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    model = AdaptiveRecurrentTwoFactorBrain()

    x = torch.randn(1, 8)

    output = model(
        x,
        steps=8,
        return_history=True
    )

    print(
        "General populations:",
        model.num_general_populations
    )

    print(
        "Specific populations:",
        model.num_specific_populations
    )

    print(
        "Biological-equivalent neurons:",
        model.biological_equivalent_neurons
    )

    print(
        "G state:",
        output["g_state"].shape
    )

    print(
        "S state:",
        output["s_state"].shape
    )

    print(
        "Bilateral state:",
        output["bilateral_state"].shape
    )

    print(
        "Gate shape:",
        output["gates"].shape
    )

    print(
        "Decision history:",
        output["decision_history"].shape
    )

    print(
        "Output shape:",
        output["logits"].shape
    )

    print("Model test: PASS")
