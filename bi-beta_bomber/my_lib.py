import os
import time
import numpy as np
import random
import tensorflow as tf
from collections import deque
from keras.models import load_model
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Concatenate
from keras.optimizers import Adam
from keras.initializers import HeNormal
from keras.callbacks import TensorBoard
# from keras import utils
import settings as s
import events as e
from .my_settings import *


# Own Tensorboard class - to save stats only once instead of at each fit
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self._should_write_train_graph = False

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        # self._write_logs(stats, self.step
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class DQNAgent:
    def __init__(self, is_training, model_path=None):
        # Create main model
        if model_path is None:
            self.model = self.create_model_advanced()
        else:
            # self.model.load_weights(model_path)
            self.model = load_model(model_path)

        if is_training:
            # Create target model - from scratch instead of loading to more easily notice possible mistakes
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

            # Replay memory
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

            # Custom tensor board
            self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, time.strftime('%Y%m%d-%H%M%S')))

            # Counter for updating target model
            self.target_update_counter = 0

    def create_model_advanced(self):
        # tf.debugging.set_log_device_placement(True)

        # ----- MULTI GPU NOT WORKING - TO DEBUG (Model input needs to be tensor data?) -----
        # if tf.config.list_physical_devices("GPU"):
        #     strategy = tf.distribute.MirroredStrategy()
        #     with strategy.scope():
        #         return self.create_model()
        # else:
        #     return self.create_model()
        return self.create_model()

    # noinspection PyMethodMayBeStatic
    def create_model(self):
        init = HeNormal()

        # DQN using Functional API - Sequential() does not support multiple input layers in different places
        # conv_input = Input(shape=(CONV_SIZE, CONV_SIZE, N_CONV_INPUTS))
        other_input = Input(shape=(OTHER_INPUTS_DIM, ))

        # conv_layer = conv_input
        # for i in range(len(CONV_FILTERS)):
        #     conv_layer = Conv2D(CONV_FILTERS[i], KERNEL_SIZES[i], padding="same", kernel_initializer=init)(conv_layer)
        #     conv_layer = Activation("relu")(conv_layer)
        #     conv_layer = MaxPooling2D(POOL_SIZES[i], strides=POOL_STRIDES[i])(conv_layer)
        #
        # flat_conv = Flatten()(conv_layer)
        # fc_layer = Concatenate()([flat_conv, other_input])
        fc_layer = other_input

        for i in range(len(DENSE_UNITS)):
            fc_layer = Dense(DENSE_UNITS[i], activation="relu", kernel_initializer=init)(fc_layer)
        output = Dense(len(ACTIONS), activation="linear")(fc_layer)

        # full_model = Model(inputs=[conv_input, other_input], outputs=output)
        full_model = Model(inputs=other_input, outputs=output)
        # utils.plot_model(full_model)
        full_model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        return full_model

    # inputs must be [conv_inputs, other_inputs] - old with conv
    def get_qs(self, inputs):
        return self.model.predict(inputs)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        # Train only if enough steps in replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Create random mini batch from replay memory
        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        # print(len(mini_batch))

        # Get current (future) states from mini_batch, then query NN model (target model) for Q values
        current_states = []
        new_current_states = []
        for transition in mini_batch:
            current_states.append(transition[0][0])
            new_current_states.append(transition[3][0])
        current_states = np.array(current_states)
        new_current_states = np.array(new_current_states)

        current_qs_list = self.model.predict(current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_states, action, reward, new_current_states, done) in enumerate(mini_batch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            x.append(current_states[0])
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(x), np.array(y),
                       batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every round
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def my_pooling_2d(data, pool_size, stride, pad_size=0):
    """
    Perform 2DmaxPooling
    - data: 2D numpy array
    - pool_size: int - pooling width and height
    - stride: int - horizontal and vertical stride
    - pad_size: int - padding to add to data
    """
    # pad data
    m = np.pad(data, pad_width=pad_size)
    # initialize output matrix
    out_dim = int(1 + (data.shape[0] - pool_size + 2*pad_size) / stride)
    output = np.zeros((out_dim, out_dim))
    # fill output matrix
    for i in range(out_dim):
        x_start = i*stride
        x_end = x_start + pool_size
        for j in range(out_dim):
            y_start = j * stride
            y_end = y_start + pool_size
            output[i, j] = np.max(m[x_start:x_end, y_start:y_end])
    return output


def map_coins(gs_coins):
    """
    create global game map of coins from games_state["coins"]
    """
    # initialize global game field
    coin_map = np.zeros((s.COLS, s.ROWS))
    # place available coins on field
    for i in range(len(gs_coins)):
        coin_map[gs_coins[i][0], gs_coins[i][1]] = 1.
    return coin_map


def map_players(gs_players):
    """
    create global game map of player(s) from games_state
    gs_players must be a list of players, i.e. for self pass [game_state["self"]]
    """
    # initialize global game field
    players_map = np.zeros((s.COLS, s.ROWS))
    # place players on field
    for i in range(len(gs_players)):
        players_map[gs_players[i][3][0], gs_players[i][3][1]] = 1.
    return players_map


def map_bombs(arena, gs_bombs):
    """
    create global game map of bombs and relative explosion tiles - radius and timer of bomb
    - arena: whole arena from game state - contains information about walls
    - gs_bombs: game_state["bombs"]
    """
    # initialize global game field
    bombs_map = np.zeros_like(arena, dtype="float")
    # for each bomb get coordinates and map explosion tiles with bomb bomb stage
    # if multiple on same tile keep one that is closer to explosion
    for i in range(len(gs_bombs)):
        x = gs_bombs[i][0][0]
        y = gs_bombs[i][0][1]
        bomb_stage = 1. - gs_bombs[i][1] / s.BOMB_TIMER
        bombs_map[x, y] = bomb_stage

        # add bomb radius
        for j in range(1, s.BOMB_POWER + 1):
            if arena[x + j, y] == -1.:
                break
            bombs_map[x + j, y] = np.max([bombs_map[x + j, y], bomb_stage])
        for j in range(1, s.BOMB_POWER + 1):
            if arena[x - j, y] == -1.:
                break
            bombs_map[x - j, y] = np.max([bombs_map[x - j, y], bomb_stage])
        for j in range(1, s.BOMB_POWER + 1):
            if arena[x, y + j] == -1.:
                break
            bombs_map[x, y + j] = np.max([bombs_map[x, y + j], bomb_stage])
        for j in range(1, s.BOMB_POWER + 1):
            if arena[x, y - j] == -1.:
                break
            bombs_map[x, y - j] = np.max([bombs_map[x, y - j], bomb_stage])
    return bombs_map


def get_local_map(self, coord, field):
    """
    Create map centered on agent with field of view range:
    - coord: coordinates tuple to center map at
    - field: map to be centred
    """
    field_adj = np.pad(field, self.offset)
    x = coord[0] + self.offset
    y = coord[1] + self.offset
    return field_adj[x-self.fov:x+self.fov+1, y-self.fov:y+self.fov+1]


def map_l_my_bomb(self, gs):
    """
    Create local map of agent's own bomb:
    - gs: game_state
    """
    if not gs["self"][2]:
        # get bomb stage of my bomb and map
        for i in range(len(gs["bombs"])):
            if gs["bombs"][i][0] == self.my_bomb_coord:
                my_bomb_map = map_bombs(gs["field"], [gs["bombs"][i]])
                return get_local_map(self, gs["self"][3], my_bomb_map)
    return np.zeros((self.grid, self.grid))


# --- DEPRECATED ---
def get_available_coins_map(self, current_coins):
    """
    keep track of not picked coins per field area, i.e. on 3x3 matrix
    """
    coin_difference = current_coins - self.prev_coins_pooled
    collected = np.where(coin_difference < 0, 0, 1)
    # collected = np.sum(coin_difference < 0)/self.tot_coins
    # return np.round(self.hidden_coins - collected, 1)
    return collected * self.hidden_coins


def get_inputs(self, game_state):
    """
    encode game_state to inputs for model
    """
    if game_state is None:
        return None

    # check if in last step bomb was dropped and in case set my_bomb_coord
    if not game_state["self"][2] and self.prev_bomb_action_state:
        self.my_bomb_coord = game_state["self"][3]
    self.prev_bomb_action_state = game_state["self"][2]

    # --- Global maps 2D inputs ---
    arena = game_state["field"].astype("float")  # tmp for other variables
    arena_blocked = np.where(arena == 0., 0., 1.)
    crates = np.where(arena == 1., 1., 0.)
    bombs = map_bombs(arena, game_state["bombs"])

    # Self pooled maps to 3x3
    coins = map_coins(game_state["coins"])
    coins_pooled = my_pooling_2d(coins[1:-1, 1:-1], 5, 5)

    others = map_players(game_state["others"])
    others_pooled = my_pooling_2d(others[1:-1, 1:-1], 5, 5)

    tmp = self.hidden_coins - coins_pooled
    self.hidden_coins = np.where(tmp <= 0, 0, 1)

    me_pooled = my_pooling_2d(map_players([game_state["self"]])[1:-1, 1:-1], 5, 5)

    # --- Local maps 2D inputs ---
    l_arena_blocked = get_local_map(self, game_state["self"][3], arena_blocked)
    l_crates = get_local_map(self, game_state["self"][3], crates)
    l_coins = get_local_map(self, game_state["self"][3], coins)
    l_others = get_local_map(self, game_state["self"][3], others)
    l_bombs = get_local_map(self, game_state["self"][3], bombs)
    l_my_bomb = map_l_my_bomb(self, game_state)

    # --- 1D inputs ---
    step = game_state["step"] / s.MAX_STEPS  # game step - max 400 converted in [0, 1] by 0.0025 increments
    my_bomb_state = float(game_state["self"][2])

    # update self
    self.prev_coins_pooled = coins_pooled

    # group inputs into np.arrays of right shape for model
    # channel first (3, 17, 17) -> channel last (17, 17, 3)
    # conv_inputs = np.moveaxis(np.array([arena_blocked, crates, bombs]), 0, 2)

    pooled_inputs = np.array([coins_pooled, self.hidden_coins, others_pooled, me_pooled]).flatten()  # (36,)

    local_map_inputs = np.hstack([np.delete(l_arena_blocked.flatten(), self.grid_centre),
                                  np.delete(l_crates.flatten(), self.grid_centre),
                                  np.delete(l_coins.flatten(), self.grid_centre),
                                  np.delete(l_others.flatten(), self.grid_centre),
                                  l_bombs.flatten(),
                                  l_my_bomb.flatten()])  # (148,)

    other_inputs = np.array([step, my_bomb_state])  # (2,)

    # if in training save rotated input states
    if self.train:
        save_rotated_maps(self, coins_pooled, self.hidden_coins, others_pooled, me_pooled,
                          l_arena_blocked, l_crates, l_coins, l_others, l_bombs, l_my_bomb,
                          other_inputs)

    other_inputs_reshaped = np.hstack([pooled_inputs, local_map_inputs, other_inputs]).reshape((1, -1))

    # return [conv_inputs_reshaped, other_inputs_reshaped]
    return other_inputs_reshaped


def save_rotated_maps(self, coins_pooled_, available_coins_, others_pooled_, me_pooled_,
                      l_arena_blocked_, l_crates_, l_coins_, l_others_, l_bombs_, l_my_bomb_,
                      other_inputs):
    """
    create rotated input states and save in self.---
    """
    coins_pooled_90 = np.rot90(coins_pooled_)
    coins_pooled_180 = np.rot90(coins_pooled_90)
    coins_pooled_270 = np.rot90(coins_pooled_180)

    available_coins_90 = np.rot90(available_coins_)
    available_coins_180 = np.rot90(available_coins_90)
    available_coins_270 = np.rot90(available_coins_180)

    others_pooled_90 = np.rot90(others_pooled_)
    others_pooled_180 = np.rot90(others_pooled_90)
    others_pooled_270 = np.rot90(others_pooled_180)

    me_pooled_90 = np.rot90(me_pooled_)
    me_pooled_180 = np.rot90(me_pooled_90)
    me_pooled_270 = np.rot90(me_pooled_180)

    l_arena_blocked_90 = np.rot90(l_arena_blocked_)
    l_arena_blocked_180 = np.rot90(l_arena_blocked_90)
    l_arena_blocked_270 = np.rot90(l_arena_blocked_180)

    l_crates_90 = np.rot90(l_crates_)
    l_crates_180 = np.rot90(l_crates_90)
    l_crates_270 = np.rot90(l_crates_180)

    l_coins_90 = np.rot90(l_coins_)
    l_coins_180 = np.rot90(l_coins_90)
    l_coins_270 = np.rot90(l_coins_180)

    l_others_90 = np.rot90(l_others_)
    l_others_180 = np.rot90(l_others_90)
    l_others_270 = np.rot90(l_others_180)

    l_bombs_90 = np.rot90(l_bombs_)
    l_bombs_180 = np.rot90(l_bombs_90)
    l_bombs_270 = np.rot90(l_bombs_180)

    l_my_bomb_90 = np.rot90(l_my_bomb_)
    l_my_bomb_180 = np.rot90(l_my_bomb_90)
    l_my_bomb_270 = np.rot90(l_my_bomb_180)

    pooled_inputs_90 = np.array([coins_pooled_90, available_coins_90, others_pooled_90, me_pooled_90]).flatten()
    pooled_inputs_180 = np.array([coins_pooled_180, available_coins_180, others_pooled_180, me_pooled_180]).flatten()
    pooled_inputs_270 = np.array([coins_pooled_270, available_coins_270, others_pooled_270, me_pooled_270]).flatten()

    local_map_inputs_90 = np.hstack([np.delete(l_arena_blocked_90.flatten(), self.grid_centre),
                                     np.delete(l_crates_90.flatten(), self.grid_centre),
                                     np.delete(l_coins_90.flatten(), self.grid_centre),
                                     np.delete(l_others_90.flatten(), self.grid_centre),
                                     l_bombs_90.flatten(),
                                     l_my_bomb_90.flatten()])

    local_map_inputs_180 = np.hstack([np.delete(l_arena_blocked_180.flatten(), self.grid_centre),
                                     np.delete(l_crates_180.flatten(), self.grid_centre),
                                     np.delete(l_coins_180.flatten(), self.grid_centre),
                                     np.delete(l_others_180.flatten(), self.grid_centre),
                                     l_bombs_180.flatten(),
                                     l_my_bomb_180.flatten()])

    local_map_inputs_270 = np.hstack([np.delete(l_arena_blocked_270.flatten(), self.grid_centre),
                                     np.delete(l_crates_270.flatten(), self.grid_centre),
                                     np.delete(l_coins_270.flatten(), self.grid_centre),
                                     np.delete(l_others_270.flatten(), self.grid_centre),
                                     l_bombs_270.flatten(),
                                     l_my_bomb_270.flatten()])

    self.inputs_90 = np.hstack([pooled_inputs_90, local_map_inputs_90, other_inputs]).reshape((1, -1))
    self.inputs_180 = np.hstack([pooled_inputs_180, local_map_inputs_180, other_inputs]).reshape((1, -1))
    self.inputs_270 = np.hstack([pooled_inputs_270, local_map_inputs_270, other_inputs]).reshape((1, -1))
    return None


def rotate_action(rot, action):
    """
    rotate action
    - rot: degrees to rotate
    - action: int - action to rotate
    """
    if action < 4:
        n = rot/90
        return int((action + n) % 4)
    else:
        return action


