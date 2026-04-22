import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    PPO attacker — adversary / jailbreak generator.

    Operates entirely in embedding space (384-d → 384-d perturbation delta).
    No text decoding is required; the action is a continuous perturbation
    vector applied directly to the frozen encoder's output.

    State  : 384-d embedding of a (possibly leet-augmented) prompt.
    Action : 384-d perturbation Δ  →  perturbed_emb = state + scale * Δ.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, output_dim)
        # Per-dimension learnable log-std; clamped to avoid numerical overflow.
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, state: torch.Tensor):
        """Sample an action and return (action, log_prob) — used during rollout."""
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(torch.clamp(self.log_std, -20, 2)).expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Re-compute log_prob and entropy for stored actions under the current
        (updated) policy.  Called inside the PPO update with gradients enabled.
        """
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(torch.clamp(self.log_std, -20, 2)).expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
