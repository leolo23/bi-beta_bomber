import os
import pickle
import random

import numpy as np

from .train import get_state_index ############ Added

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Initial = True
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
            self.Initial = False
    
    self.Explore = False

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    self.logger.debug("Querying model for action.")
    
    if self.Initial == True:
        self.Initial = False
        return np.random.choice(ACTIONS)  # Initial step
    else:
        if game_state['coins'] == []:
            return 'WAIT'
        else:
            ## Reorder List of coins, s.t. closest coin is first element
            #dist = np.linalg.norm(np.array(game_state['self'][-1])-np.array(game_state['coins']),axis=-1)
            #closest_coin_ind = np.argmin(dist)
            #game_state['coins'][0], game_state['coins'][closest_coin_ind] = game_state['coins'][closest_coin_ind], game_state['coins'][0]


            if self.Explore == True:
                rand_action = np.random.choice(ACTIONS)
                self.logger.info(f"Random act(): action: {rand_action}")
                self.Explore = False
                return rand_action  # Explore environment   
            else:
                action = np.argmax(self.q_table[get_state_index(self,game_state),:])
                self.logger.info(f"Else statement in act(): action: {action}")
                return ACTIONS[action]   # Exploit environment

