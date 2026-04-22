# ==============================================================================
# config.py — Hyperparameters
# ==============================================================================

# ── Encoder ───────────────────────────────────────────────────────────────────
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ENCODER_MAX_LEN = 128

# ── Training loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 100
PROMPTS_PER_EPOCH = 10          # queries per epoch; PPO updated after each batch
DEFENDER_STEPS_PER_EPOCH = 3    # gradient steps on augmented buffer per epoch

# ── Replay buffer ─────────────────────────────────────────────────────────────
REPLAY_BUFFER_MAX = 2_000       # FIFO capacity

# ── Attacker (PPO) ────────────────────────────────────────────────────────────
PERTURBATION_SCALE = 0.1        # delta = state + PERTURBATION_SCALE * actor_output
PPO_EPOCHS = 10                 # update epochs per PPO step
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEF = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

# ── Defender (MLP) ───────────────────────────────────────────────────────────
DEFENDER_LR = 1e-3

# ── Weighted Focal BCE Loss ───────────────────────────────────────────────────
FOCAL_W_HARM = 30.0             # weight for harmful class (label=0)
FOCAL_W_SAFE = 1.0              # weight for safe class    (label=1)
FOCAL_GAMMA = 2.0               # focal exponent

# ── Data augmentation ─────────────────────────────────────────────────────────
LEET_RATE = 0.3                 # probability of leet-transforming a prompt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
