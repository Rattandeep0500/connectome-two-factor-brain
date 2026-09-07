import torch
import torch.nn as nn


class GenericRecurrentBrain(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        n_tasks: int = 18,
        task_embedding_dim: int = 16,
        rollout_steps: int = 8,
        output_dim: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.task_embedding_dim = task_embedding_dim
        self.rollout_steps = rollout_steps
        self.output_dim = output_dim

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.task_embedding = nn.Embedding(
            n_tasks,
            task_embedding_dim,
        )

        self.context_projection = nn.Linear(
            hidden_dim + task_embedding_dim,
            hidden_dim,
        )

        self.recurrent = nn.GRUCell(
            hidden_dim,
            hidden_dim,
        )

        self.decision = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.confidence = nn.Linear(
            hidden_dim,
            1,
        )

        self.state_history = []

    def reset_history(self):
        self.state_history = []

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        x: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")

        if task_index.ndim != 1:
            raise ValueError("task_index must have shape [batch]")

        if x.shape[0] != task_index.shape[0]:
            raise ValueError("Batch dimensions must match")

        self.reset_history()

        base = torch.tanh(
            self.input_projection(x)
        )

        task_context = self.task_embedding(task_index)

        context = torch.tanh(
            self.context_projection(
                torch.cat(
                    [base, task_context],
                    dim=-1,
                )
            )
        )

        state = torch.zeros_like(context)

        for _ in range(self.rollout_steps):
            state = self.recurrent(
                context,
                state,
            )
            self.state_history.append(state)

        logits = self.decision(state)
        confidence = torch.sigmoid(
            self.confidence(state)
        )

        history = torch.stack(
            self.state_history,
            dim=1,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence,
            "state": state,
            "history": history,
        }


if __name__ == "__main__":
    torch.manual_seed(20260907)

    model = GenericRecurrentBrain()

    x = torch.randn(32, 16)
    task_index = torch.randint(
        0,
        18,
        (32,),
    )

    output = model(
        x,
        task_index,
    )

    print(f"Parameters: {model.parameter_count()}")
    print(f"Logits: {output['logits'].shape}")
    print(f"Probabilities: {output['probabilities'].shape}")
    print(f"Confidence: {output['confidence'].shape}")
    print(f"Recurrent history: {output['history'].shape}")

    assert output["logits"].shape == (32, 2)
    assert output["probabilities"].shape == (32, 2)
    assert output["confidence"].shape == (32, 1)
    assert output["history"].shape == (32, 8, 64)

    print("Generic recurrent model test: PASS")
