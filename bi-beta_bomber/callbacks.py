# import os
# import sys
# import numpy as np
# import settings as s
# import pickle
from .my_lib import *
from .my_settings import *


# ----- Agent -----
def setup(self):

    # Load or create model
    file_path_in_dir = os.path.join(MODELS_DIR, MY_MODEL)
    if os.path.isfile(file_path_in_dir):
        self.agent = DQNAgent(self.train, file_path_in_dir)
        # print("\n\nLoaded model", file_path_in_dir)
        self.logger.info("Loading model")
    elif os.path.isfile(MY_MODEL):
        self.agent = DQNAgent(self.train, MY_MODEL)
        # print("\n\nLoaded model", MY_MODEL)
        self.logger.info("Loading model")
    else:
        self.agent = DQNAgent(self.train)
        # print("\n\nSetting up model from scratch")
        self.logger.info("Setting up model from scratch.")

    # print(self.agent.model.summary())
    # print("\n")
    # print("Num CPUs:", len(tf.config.list_physical_devices("CPU")))
    # print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))
    # print("\n")

    # set agent variables for inputs and tracking
    self.fov = AGENT_FIELD_OF_VIEW
    self.grid = self.fov*2 + 1  # grid side length
    self.offset = self.fov - 1
    self.grid_centre = int(np.floor(self.grid**2 / 2))

    self.prev_bomb_action_state = True
    self.my_bomb_coord = tuple()
    self.prev_coins_pooled = np.zeros((3, 3))
    self.hidden_coins = np.ones((3, 3))   # ADJUST FOR SIMPLIFIED ENVIRONMENT Stage 1 training

    self.logger.info("Agent initialized")
    self.current_round = 0

    self.t_start_round = time.time()


def reset_self(self):
    # print("Rest self")
    self.my_bomb_coord = tuple()  # maybe there is a better way
    self.prev_coins_pooled = np.zeros((3, 3))
    self.hidden_coins = np.ones((3, 3))  # ADJUST FOR SIMPLIFIED ENVIRONMENT Stage 1 training

    self.t_start_round = time.time()

    if self.train:
        self.current_round_reward = 0
        if not (self.current_round - 1) % RESET_EPSILON_EVERY:
            self.epsilon = EPSILON
        self.position_history = np.zeros((17, 17))
        self.step_bomb_unused = 0


def act(self, game_state):

    # print("Step:", game_state["step"])
    # rest self at new round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    # get inputs and action
    if not self.train:
        state_inputs = get_inputs(self, game_state)
        action = np.argmax(self.agent.get_qs(state_inputs))
    else:
        if game_state["step"] == 1:
            state_inputs = get_inputs(self, game_state)
            self.state_inputs = state_inputs
        else:
            state_inputs = self.state_inputs

        if np.random.random() > self.epsilon:
            action = np.argmax(self.agent.get_qs(state_inputs))
        else:
            # Random action
            action = np.random.randint(0, len(ACTIONS))
            # increased probability to use bomb - stage 3 training
            # if game_state["self"][2] and np.random.random() > 0.6:
            #     action = 5
        # check if bomb can be used
        if action == 5 and not CAN_USE_BOMB:
            # Random action - last action (5) is BOMB
            action = np.random.randint(0, len(ACTIONS) - 1)

        self.action = action

    self.logger.info(ACTIONS[action])

    return ACTIONS[action]
