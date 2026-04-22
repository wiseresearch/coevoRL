import torch


def get_reward(
    original_emb: torch.Tensor,
    perturbed_emb: torch.Tensor,
    original_label: int,
    defender,
) -> tuple[float, str, float]:
    """
    Compute the attacker's reward for a single (state, action) pair.

    Defender query (embedding space, no LLM oracle):
        label=0 (harmful): +1 if Defender predicts "safe"  (jailbreak success)
                           −1 if Defender predicts "unsafe" (defense wins)
        label=1 (safe)   :  0  calibration; not an attack scenario.

    Parameters
    ----------
    original_emb  : (1, 384) tensor — unperturbed embedding (unused, kept for API compat).
    perturbed_emb : (1, 384) tensor — embedding after attacker perturbation.
    original_label: int — ground-truth label (0 = harmful, 1 = safe).
    defender      : Defender model in eval mode.

    Returns
    -------
    reward   : float
    outcome  : str  — one of JAILBREAK_SUCCESS | DEFENSE_SUCCESS | CALIBRATION_SAFE
    prob_safe: float — P(safe) from Defender
    """
    defender.eval()
    with torch.no_grad():
        prob_safe = defender.predict_proba(perturbed_emb).item()

    if original_label == 0:
        reward = 1.0 if prob_safe >= 0.5 else -1.0
        outcome = "JAILBREAK_SUCCESS" if prob_safe >= 0.5 else "DEFENSE_SUCCESS"
    else:
        reward = 0.0
        outcome = "CALIBRATION_SAFE"

    return reward, outcome, prob_safe


def weighted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_harm: float,
    w_safe: float,
    gamma: float,
) -> torch.Tensor:
    """
    Weighted Focal BCE Loss for the Defender.

    Formula (per sample i):
        p_t  = σ(logit)  if target==1  else  1 − σ(logit)
        w_i  = w_safe    if target==1  else  w_harm
        L_i  = w_i · (1 − p_t)^γ · (−log p_t)
        Loss = mean(L_i)

    Properties
    ----------
    • w_harm >> w_safe   → gradient dominated by false-negative correction.
    • Focal term         → near-zero gradient for easy examples; full gradient
                           for hard/uncertain ones (dynamic per-batch).

    Parameters
    ----------
    logits  : raw Defender output (no Sigmoid applied).
    targets : float tensor, 1 = safe, 0 = harmful.
    w_harm  : class weight for the harmful class.
    w_safe  : class weight for the safe class.
    gamma   : focal exponent.
    """
    probs = torch.sigmoid(logits)
    p_t = torch.where(targets == 1, probs, 1.0 - probs)
    weights = torch.where(
        targets == 1,
        torch.full_like(targets, w_safe),
        torch.full_like(targets, w_harm),
    )
    bce = -torch.log(p_t.clamp(min=1e-8))
    focal = weights * (1.0 - p_t) ** gamma * bce
    return focal.mean()
