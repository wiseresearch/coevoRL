import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    PPO critic — estimates the expected return V(s).

    Used as a baseline to compute advantages in the PPO update:
        advantage_i = reward_i - V(state_i)

    Trained with MSE loss against normalised observed rewards.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
