import torch
import torch.nn as nn


class Defender(nn.Module):
    """
    Binary MLP classifier — detector / policy.

    Predicts whether a 384-d embedding belongs to the safe (1) or harmful (0)
    class.  Outputs raw logits; Sigmoid is applied externally by the loss
    function during training and by predict_proba() at inference time.

    Trained with Weighted Focal BCE:
        w_harm = 30, w_safe = 1, γ = 2  (per paper ablation).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),          # raw logit — no Sigmoid here
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def predict_proba(self, state: torch.Tensor) -> torch.Tensor:
        """Return P(safe) ∈ (0, 1)."""
        return torch.sigmoid(self.forward(state))
