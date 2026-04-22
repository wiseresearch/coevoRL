# CoEvol-Defense

Co-evolutionary adversarial training framework for LLM jailbreak detection.
Implements the method described in the QRS paper: an attacker (PPO agent) and a defender (MLP classifier) co-evolve in embedding space — no LLM oracle required at training time.

---

## Architecture

```
Frozen Encoder (all-MiniLM-L6-v2, 384-d)
        │
        ▼
  ┌─────────────┐        PPO rollout         ┌──────────────────┐
  │   Attacker  │ ──── Δ perturbation ──────▶│    Reward Fn     │
  │  (Actor-    │                            │  Phase 1: Defender│
  │   Critic)   │◀─── reward signal ─────── │  Phase 2: cos-sim │
  └─────────────┘                            └──────────────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │    Defender      │
                                            │  (MLP classifier)│
                                            │  Focal BCE loss  │
                                            └──────────────────┘
```

| Component | Role |
|-----------|------|
| **Attacker** (PPO Actor-Critic) | Generates embedding-space perturbations Δ to fool the Defender |
| **Defender** (MLP binary classifier) | Detects harmful prompts; trained on original + adversarial embeddings |
| **Reward** | +1 jailbreak success / −1 defense wins + cosine-similarity penalty (Phase 2) |
| **Focal BCE** | `w_harm=30, w_safe=1, γ=2` — suppresses false negatives on harmful class |

---

## Quickstart

```bash
pip install -r requirements.txt

python run_experiment.py \
    --data  path/to/dataset.csv \
    --epochs 100 \
    --output runs/my_experiment \
    --tb
```

### Dataset format

A CSV with at minimum two columns:

| Column  | Type | Description |
|---------|------|-------------|
| `prompt` | str  | Raw prompt text |
| `label`  | int  | `1` = safe, `0` = harmful |

---

## Project layout

```
coevo-defense/
├── README.md
├── requirements.txt
├── run_experiment.py        # single entry point
├── config.py                # all hyperparameters (paper values)
├── models/
│   ├── __init__.py
│   ├── attacker.py          # PPO Actor — embedding-space perturbation generator
│   ├── defender.py          # MLP binary classifier
│   └── critic.py            # PPO Critic — value estimator
├── training/
│   ├── __init__.py
│   └── coevolution.py       # core co-evolutionary loop + PPO update
├── env/
│   └── reward.py            # reward signal + weighted focal loss
└── utils/
    └── seed.py               # reproducibility
```

---

## Key hyperparameters (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_EPOCHS` | 100 | Co-evolution epochs |
| `PROMPTS_PER_EPOCH` | 10 | Queries per epoch (5 harmful + 5 safe) |
| `PERTURBATION_SCALE` | 0.1 | Embedding perturbation magnitude |
| `SIM_THRESHOLD` | 0.85 | Min cosine similarity (Phase 2 constraint) |
| `SIM_PENALTY_WEIGHT` | 0.5 | Phase 2 penalty coefficient |
| `FOCAL_W_HARM` | 30.0 | Focal loss weight for harmful class |
| `FOCAL_GAMMA` | 2.0 | Focal loss exponent |
| `REPLAY_BUFFER_MAX` | 2 000 | FIFO replay buffer capacity |

---

## TensorBoard

```bash
tensorboard --logdir runs/my_experiment/tensorboard
```

Tracked scalars: `Defender/{Loss,Accuracy,Recall_Harmful}`, `RL_Agent/{Actor_Loss,Critic_Loss,Average_Reward}`, `Co-Evolution/{Attack_Success_Rate,Avg_Cosine_Similarity}`.

