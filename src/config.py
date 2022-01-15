# Network
MID_1_SIZE = 128
MID_2_SIZE = 64

# Hyperparams

LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 1e-3
EPS_MAX = 1
EPS_MIN = 0.01

MAX_EPOCH = 2000
EXPLORATORY_EPOCHS = 250 # epochs during which epsilon decreases linearly

# Learning
BATCH_SIZE = 64
UPDATE_INTERVAL = 4

# Optimizer
MOMENTUM = 0.95

# Memory
BUFFER_SIZE = int(1e6)

# Paths
UNITY_ENV_LOCATION = "../Banana.app"
CHECKPOINT_SAVE_PATH = "../checkpoints/model_checkpoint.pth"


