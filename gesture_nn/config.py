"""
Configuration for DBEW-NN gesture recognition pipeline.
All tunable parameters in one place for reproducibility.
"""

# ── Data Generation ──────────────────────────────────────────────
# Simulated Rokid UXR hand skeleton: 26 joints × 3 coords (x,y,z in meters)
N_JOINTS = 26
JOINT_DIMS = 3
N_GESTURE_CLASSES = 11  # 11 gesture classes (no NONE)
FPS = 60
GESTURE_DURATION_SEC = 2.0  # typical gesture hold duration
WINDOW_SIZE = 32  # frames per input window
WINDOW_STRIDE = 4  # stride for sliding window during training

# Gesture class mapping (matching thesis Table tab:ges_map)
GESTURE_MAP = {
    0:  "fist",                # 握拳
    1:  "index_left",          # 单指向左
    2:  "index_right",         # 单指向右
    3:  "two_finger_palm",     # 二指手心
    4:  "two_finger_back",     # 二指手背
    5:  "three_finger_palm",   # 三指手心
    6:  "three_finger_back",   # 三指手背
    7:  "four_finger_palm",    # 四指手心
    8:  "four_finger_back",    # 四指手背
    9:  "five_finger_palm",    # 五指手心
    10: "five_finger_back",    # 五指手背
}

# Number of participants for synthetic data
N_PARTICIPANTS = 10
N_SESSIONS = 3
N_REPETITIONS = 10

# Data split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ── Model Architecture ───────────────────────────────────────────
D_MODEL = 96          # hidden dimension (increased from 64 for palm/back discrimination)
N_HEADS = 6           # self-attention heads (head_dim = 96/6 = 16)
N_CNN_LAYERS = 3      # 1D-CNN layers
CNN_KERNEL = 3        # CNN kernel size
DILATIONS = [1, 2, 4] # dilated CNN for multi-scale temporal receptive field
DROPOUT = 0.1

# ── Training ─────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 120        # extended for harder palm/back convergence
WARMUP_EPOCHS = 8     # longer warmup to stabilise CNN before attention unfreeze
EARLY_STOP_PATIENCE = 15

# Loss weights
TEMPORAL_SMOOTH_LAMBDA = 0.05  # reduced to allow sharper palm/back discrimination

# Data augmentation
AUG_TIME_CROP_JITTER = 5   # ±5 frames random crop
AUG_COORD_NOISE_SIGMA = 0.002  # 2mm gaussian noise (meters)
AUG_MIRROR_PROB = 0.2      # reduced: mirror can blur palm/back z-axis signal

# Class balancing
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 3.0           # stronger focus on hard samples (palm/back pairs)

# ── DBEW Trigger Pipeline (matching thesis Section 3.3.4) ────────
TAU_MS = 500           # cooldown window (ms)
K_MIN = 8              # minimum stable frames
THETA_G = 0.85         # confidence threshold for NN softmax output
THETA_EXT = 0.85  # collinearity threshold for rule-based (≈32° max angular deviation)

# ── Export ───────────────────────────────────────────────────────
ONNX_OPSET = 14
EXPORT_INPUT_NAME = "skeleton_sequence"  # [1, 32, 26, 3]
EXPORT_OUTPUT_NAME = "gesture_logits"    # [1, 11]
