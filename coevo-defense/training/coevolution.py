"""
training/coevolution.py — Core co-evolutionary training loop.

Architecture:
    Frozen encoder  all-MiniLM-L6-v2  →  384-d embeddings (never updated)

    Attacker (PPO Actor-Critic) in embedding space:
        state         = 384-d embedding of a prompt
        action        = 384-d perturbation vector Δ
        perturbed_emb = state + PERTURBATION_SCALE × Δ
        reward        = Defender-based (no LLM oracle)

    Defender (MLP binary classifier):
        Loss = Weighted Focal BCE  (w_h=30, w_l=1, γ=2)
        Trains on original data + adversarial examples (co-evolution step)

    Co-evolution loop (100 epochs, 10 prompts/epoch):
        Attack Phase  → PPO update every 10 queries
        Defense Phase → 3 Defender gradient steps on augmented FIFO buffer
"""

import random

import numpy as np
import torch
import torch.nn as nn

from config import (
    EMBEDDING_DIM,
    FOCAL_GAMMA,
    FOCAL_W_HARM,
    FOCAL_W_SAFE,
    LEET_RATE,
    NUM_EPOCHS,
    PERTURBATION_SCALE,
    PPO_CLIP_EPSILON,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PROMPTS_PER_EPOCH,
    DEFENDER_STEPS_PER_EPOCH,
    REPLAY_BUFFER_MAX,
)
from env.reward import get_reward, weighted_focal_loss


# ---------------------------------------------------------------------------
# Leet augmentation helper
# ---------------------------------------------------------------------------
def _to_leet(text: str, substitution_rate: float = 0.5) -> str:
    """Probabilistic character-level leet substitution."""
    leet_map = {"a": "@", "b": "8", "e": "3", "o": "0", "s": "$", "t": "7"}
    return "".join(
        leet_map[c] if c in leet_map and random.random() < substitution_rate else c
        for c in text.lower()
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def _update_ppo(
    memory: dict,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device: torch.device,
    epochs: int = PPO_EPOCHS,
    clip_epsilon: float = PPO_CLIP_EPSILON,
    entropy_coef: float = PPO_ENTROPY_COEF,
) -> tuple[float, float]:
    """
    Standard PPO update:
        • Reward normalisation          (stabilises gradient scale)
        • Advantage normalisation       (reduces variance)
        • Separate actor / critic steps (avoids cross-contaminated graphs)
        • Gradient clipping             (prevents exploding gradients)
        • Entropy bonus                 (encourages exploration)
    """
    states = torch.stack(memory["states"]).to(device).detach()
    actions = torch.stack(memory["actions"]).to(device).detach()
    old_log_probs = torch.stack(memory["log_probs"]).to(device).detach()
    rewards = torch.tensor(memory["rewards"], dtype=torch.float32).to(device)

    if rewards.std() > 1e-8:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    actor_loss_val = critic_loss_val = 0.0

    for _ in range(epochs):
        # ── Actor update ──────────────────────────────────────────────────
        actor_optimizer.zero_grad()
        new_log_probs, entropy = actor.evaluate(states, actions)

        with torch.no_grad():
            advantages = rewards - critic(states).squeeze()
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        actor_optimizer.step()

        # ── Critic update ─────────────────────────────────────────────────
        critic_optimizer.zero_grad()
        values = critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, rewards)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        critic_optimizer.step()

        actor_loss_val = actor_loss.item()
        critic_loss_val = critic_loss.item()

    return actor_loss_val, critic_loss_val


# ---------------------------------------------------------------------------
# CoEvolutionTrainer
# ---------------------------------------------------------------------------
class CoEvolutionTrainer:
    """
    Orchestrates the full co-evolutionary training loop.

    Parameters
    ----------
    actor, critic, defender : model instances (already on the correct device).
    actor_optimizer, critic_optimizer, defender_optimizer : torch optimizers.
    prompt_embeddings : np.ndarray of shape (N, 384) — pre-computed frozen embeddings.
    labels            : list[int] — ground-truth labels (0=harmful, 1=safe).
    prompts           : list[str] — raw prompt strings (used for leet augmentation).
    get_embedding     : callable(str) -> np.ndarray — encoder function.
    device            : torch.device.
    writer            : optional SummaryWriter for TensorBoard logging.
    """

    def __init__(
        self,
        actor,
        critic,
        defender,
        actor_optimizer,
        critic_optimizer,
        defender_optimizer,
        prompt_embeddings: np.ndarray,
        labels: list,
        prompts: list,
        get_embedding,
        device: torch.device,
        writer=None,
    ):
        self.actor = actor
        self.critic = critic
        self.defender = defender
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.defender_opt = defender_optimizer
        self.prompt_embeddings = prompt_embeddings
        self.labels = labels
        self.prompts = prompts
        self.get_embedding = get_embedding
        self.device = device
        self.writer = writer

        self.safe_indices = [i for i, l in enumerate(labels) if l == 1]
        self.unsafe_indices = [i for i, l in enumerate(labels) if l == 0]
        if not self.safe_indices or not self.unsafe_indices:
            raise ValueError("Dataset must contain both safe (label=1) and unsafe (label=0) prompts.")

        # Seed the replay buffer with the original dataset
        self.replay_embeddings: list = list(prompt_embeddings.copy())
        self.replay_labels: list = list(labels)

        self.training_logs: list[dict] = []

    # ------------------------------------------------------------------
    def train(self, num_epochs: int = NUM_EPOCHS) -> list[dict]:
        """
        Run the co-evolutionary loop for `num_epochs` epochs.

        Returns
        -------
        training_logs : list of per-step dicts with keys:
            epoch, original_label, outcome, prob_safe, reward, is_leet
        """
        for epoch in range(num_epochs):
            print(f"\n{'─'*60}")
            print(f"  EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'─'*60}")

            ppo_memory: dict = {"states": [], "actions": [], "log_probs": [], "rewards": []}

            # Balanced sampling: half unsafe (attack targets), half safe (calibration)
            n_unsafe = PROMPTS_PER_EPOCH // 2
            n_safe = PROMPTS_PER_EPOCH - n_unsafe
            combined = list(
                np.random.choice(self.unsafe_indices, n_unsafe, replace=True)
            ) + list(
                np.random.choice(self.safe_indices, n_safe, replace=True)
            )
            random.shuffle(combined)

            new_embs: list = []
            new_labels: list = []

            self.actor.eval()

            for idx in combined:
                original_label = self.labels[idx]
                prompt_text = self.prompts[idx]

                # Optional leet augmentation
                is_leet = random.random() < LEET_RATE
                if is_leet:
                    prompt_text = _to_leet(prompt_text)
                    base_emb = self.get_embedding(prompt_text)
                else:
                    base_emb = self.prompt_embeddings[idx]

                state = torch.tensor(base_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    perturbation, log_prob = self.actor(state)

                perturbed_emb = state + PERTURBATION_SCALE * perturbation

                reward, outcome, prob_safe = get_reward(
                    state, perturbed_emb, original_label, self.defender
                )

                ppo_memory["states"].append(state.squeeze(0))
                ppo_memory["actions"].append(perturbation.squeeze(0).detach())
                ppo_memory["log_probs"].append(log_prob.squeeze(0).detach())
                ppo_memory["rewards"].append(reward)

                new_embs.append(perturbed_emb.squeeze(0).detach().cpu().numpy())
                new_labels.append(original_label)

                tag = "ATTACK" if original_label == 0 else "CAL"
                print(
                    f"  [{tag}] {outcome:<22}  "
                    f"P(safe)={prob_safe:.3f}  r={reward:+.3f}"
                    + ("  [leet]" if is_leet else "")
                )

                self.training_logs.append(
                    {
                        "epoch": epoch + 1,
                        "original_label": original_label,
                        "outcome": outcome,
                        "prob_safe": prob_safe,
                        "reward": reward,
                        "is_leet": is_leet,
                    }
                )

            # ── PPO update (attacker) ──────────────────────────────────────
            self.actor.train()
            actor_loss, critic_loss = _update_ppo(
                ppo_memory,
                self.actor,
                self.critic,
                self.actor_opt,
                self.critic_opt,
                self.device,
            )

            # ── FIFO replay buffer update ──────────────────────────────────
            self.replay_embeddings.extend(new_embs)
            self.replay_labels.extend(new_labels)
            if len(self.replay_embeddings) > REPLAY_BUFFER_MAX:
                overflow = len(self.replay_embeddings) - REPLAY_BUFFER_MAX
                self.replay_embeddings = self.replay_embeddings[overflow:]
                self.replay_labels = self.replay_labels[overflow:]

            # ── Defender supervised update (co-evolutionary step) ──────────
            X_train = torch.tensor(
                np.array(self.replay_embeddings), dtype=torch.float32
            ).to(self.device)
            y_train = torch.tensor(
                np.array(self.replay_labels), dtype=torch.float32
            ).view(-1, 1).to(self.device)

            self.defender.train()
            d_loss_val = 0.0
            for _ in range(DEFENDER_STEPS_PER_EPOCH):
                self.defender_opt.zero_grad()
                logits = self.defender(X_train)
                d_loss = weighted_focal_loss(
                    logits, y_train, FOCAL_W_HARM, FOCAL_W_SAFE, FOCAL_GAMMA
                )
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.defender.parameters(), max_norm=1.0)
                self.defender_opt.step()
                d_loss_val = d_loss.item()
            self.defender.eval()

            # ── Epoch metrics ──────────────────────────────────────────────
            with torch.no_grad():
                eval_logits = self.defender(X_train)
                eval_preds = (torch.sigmoid(eval_logits) >= 0.5).float()
                epoch_acc = (eval_preds == y_train).float().mean().item()
                harmful_mask = (y_train == 0).squeeze()
                if harmful_mask.any():
                    epoch_recall = (
                        eval_preds.squeeze()[harmful_mask] == 0
                    ).float().mean().item()
                else:
                    epoch_recall = float("nan")

            avg_reward = float(np.mean(ppo_memory["rewards"]))
            print(f"\n  Defender Loss   : {d_loss_val:.4f}")
            print(f"  Defender Acc    : {epoch_acc:.4f}  |  Recall(harm): {epoch_recall:.4f}")
            print(f"  Actor Loss      : {actor_loss:.4f}")
            print(f"  Critic Loss     : {critic_loss:.4f}")
            print(f"  Avg Reward      : {avg_reward:+.4f}")

            # ── TensorBoard (optional) ─────────────────────────────────────
            if self.writer is not None:
                self.writer.add_scalar("Defender/Loss",           d_loss_val,    epoch + 1)
                self.writer.add_scalar("Defender/Accuracy",       epoch_acc,     epoch + 1)
                self.writer.add_scalar("Defender/Recall_Harmful", epoch_recall,  epoch + 1)
                self.writer.add_scalar("RL_Agent/Actor_Loss",     actor_loss,    epoch + 1)
                self.writer.add_scalar("RL_Agent/Critic_Loss",    critic_loss,   epoch + 1)
                self.writer.add_scalar("RL_Agent/Average_Reward", avg_reward,    epoch + 1)

                ep_logs = self.training_logs[-PROMPTS_PER_EPOCH:]
                ep_attacks = [l for l in ep_logs if l["original_label"] == 0]
                if ep_attacks:
                    asr = sum(
                        1 for l in ep_attacks if l["outcome"] == "JAILBREAK_SUCCESS"
                    ) / len(ep_attacks)
                    self.writer.add_scalar(
                        "Co-Evolution/Attack_Success_Rate", asr, epoch + 1
                    )

        print("\nTraining complete.")
        return self.training_logs
