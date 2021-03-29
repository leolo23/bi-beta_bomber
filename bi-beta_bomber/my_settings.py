import numpy as np
import events as e
import time

# Working Directory during execution is my_agent_dir

CAN_USE_BOMB = True
# --- SET STARTING MODEL TO LOAD --- can be in CWD or ./models
MY_MODEL = "bi-beta_bomber.h5"

# --- SET MODEL NAME --- For saving new models during training - based on settings
MODEL_NAME = "bi_beta_256-128_S4_R7_4"

# Add additional information/notes for this model, which are saved in the settings log
ADDITIONAL_NOTES = []

# NN architecture - lists for various layers
OTHER_INPUTS_DIM = 184  # dimension of one dimensional inputs to fc layer (flattened)
DENSE_UNITS = [256, 128]

AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -10_000
EPSILON = 0.3
EPSILON_DECAY = 0.986  # 0.992 reset every 1000 - 0.986 reset every 500
MIN_EPSILON = 0.001
RESET_EPSILON_EVERY = 500
DISCOUNT = 0.99

LEARNING_RATE = 0.001
UPDATE_TARGET_EVERY = 10  # rounds
REPLAY_MEMORY_SIZE = 100_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINI_BATCH_SIZE = 32

AGENT_FIELD_OF_VIEW = 2

# Fix settings
MODELS_DIR = "models"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Save time when training of model started; to be stored in settings
TIME_OF_CREATION = time.strftime("%Y-%m-%d--%H-%M-%S")

# --- TRAINING REWARDS SECTION ---

# additional events
EVENT_NO_CLOSER_TO_COIN = "NO_CLOSER_TO_COIN"
EVENT_NOT_FLEEING_OWN_BOMB = "EVENT_NOT_FLEEING_OWN_BOMB"
EVENT_STEP_LOOPING = "EVENT_STEP_LOOPING"
EVENT_NOT_USING_BOMB = "EVENT_NOT_USING_BOMB"
EVENT_NO_CLOSER_TO_NEAR_COIN = "EVENT_NO_CLOSER_TO_NEAR_COIN"
EVENT_NO_CLOSER_TO_AVAILABLE_COIN = "EVENT_NO_CLOSER_TO_AVAILABLE_COIN"
EVENT_NO_CLOSER_TO_HIDDEN_COIN = "EVENT_NO_CLOSER_TO_HIDDEN_COIN"
EVENT_NO_CLOSER_TO_ENEMIES = "EVENT_NO_CLOSER_TO_ENEMIES"

# --- UPDATE ----- if new event is added or discarded, to store information in settings -----
ADDITIONAL_EVENTS = [EVENT_NO_CLOSER_TO_COIN, EVENT_NOT_FLEEING_OWN_BOMB, EVENT_STEP_LOOPING, EVENT_NOT_USING_BOMB,
                     EVENT_NO_CLOSER_TO_NEAR_COIN, EVENT_NO_CLOSER_TO_AVAILABLE_COIN, EVENT_NO_CLOSER_TO_HIDDEN_COIN,
                     EVENT_NO_CLOSER_TO_ENEMIES]

# Define game rewards
GAME_REWARDS = {
    e.COIN_COLLECTED: 200,
    e.OPPONENT_ELIMINATED: 500,
    e.KILLED_SELF: -150,
    e.GOT_KILLED: -300,
    e.INVALID_ACTION: -50,
    # e.WAITED: -10,
    # EVENT_NO_CLOSER_TO_COIN: -10,
    EVENT_NOT_FLEEING_OWN_BOMB: -50,
    e.CRATE_DESTROYED: 50,
    e.COIN_FOUND: 100,
    # e.BOMB_DROPPED: 50
    EVENT_STEP_LOOPING: -20,
    EVENT_NOT_USING_BOMB: -20,
    EVENT_NO_CLOSER_TO_NEAR_COIN: -40,
    EVENT_NO_CLOSER_TO_AVAILABLE_COIN: -30,
    EVENT_NO_CLOSER_TO_HIDDEN_COIN: -20,
    EVENT_NO_CLOSER_TO_ENEMIES: -30
}

SETTINGS_DIR = "settings"
SETTINGS_FILE_NAME = f"settings__{MODEL_NAME}__{TIME_OF_CREATION}"

SETTINGS_INF = [TIME_OF_CREATION, MODEL_NAME, MODELS_DIR, OTHER_INPUTS_DIM, DENSE_UNITS, AGGREGATE_STATS_EVERY, MIN_REWARD, EPSILON, EPSILON_DECAY,
                MIN_EPSILON, RESET_EPSILON_EVERY, DISCOUNT, LEARNING_RATE, UPDATE_TARGET_EVERY, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE,
                MINI_BATCH_SIZE, AGENT_FIELD_OF_VIEW, ACTIONS, ADDITIONAL_EVENTS, GAME_REWARDS, ADDITIONAL_NOTES]
SETTINGS_Variables = ["TIME_OF_CREATION", "MODEL_NAME", "MODELS_DIR", "OTHER_INPUTS_DIM", "DENSE_UNITS", "AGGREGATE_STATS_EVERY", "MIN_REWARD", "EPSILON",
                      "EPSILON_DECAY", "MIN_EPSILON", "RESET_EPSILON_EVERY", "DISCOUNT", "LEARNING_RATE", "UPDATE_TARGET_EVERY", "REPLAY_MEMORY_SIZE",
                      "MIN_REPLAY_MEMORY_SIZE", "MINI_BATCH_SIZE", "AGENT_FIELD_OF_VIEW", "ACTIONS", "ADDITIONAL_EVENTS", "GAME_REWARDS", "ADDITIONAL_NOTES"]

# coordinates of tiles of pooled map in global map as coordinates of centre tile of window in global map
POOLED_CENTRE_COORD = np.array([(3, 3), (3, 8), (3, 13),
                                (8, 3), (8, 8), (8, 13),
                                (13, 3), (8, 13), (13, 13)])


# N_CONV_INPUTS = 3  # number of channels of input to conv layer
# CONV_SIZE = 17  # input size - square matrix
# CONV_FILTERS = [64]
# KERNEL_SIZES = [(3, 3)]
# POOL_SIZES = [(5, 5)]
# POOL_STRIDES = [4]
