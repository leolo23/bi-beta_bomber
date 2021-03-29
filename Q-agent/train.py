import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e

import numpy as np      ################### Added



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']  ################ Addded

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Set up and initialize q-table
    action_space_size = 6
    state_space_size = np.int(175*176)    # size of grid (free tiles): 176, number of coins: 1


    self.learning_rate = 0.01
    self.discount_rate = 0.99

    self.exploration_rate = 0
    self.min_exploration_rate = 0.01        # At least 1% exploration
    self.exploration_decay_rate = 0.995
    
    self.num_episode = 0

    self.Initial = True   # Used for first action in first round
    self.Explore = False

    # Ask if User wants to overwrite old model, or train on old model
    answer = input("Do you want to use old model or create a new one and overwrite old model? (y/n): ")
    if answer != 'y' and answer != 'n':
        raise Exception("Invalid input.  Try again...(y/n)") 
    if answer == 'y':
        print("Training on old model!")
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
            print(f"Shape of q_table: {self.q_table.shape}")
            # if self.q_table.shape[0] == 30800:
            #     print("Dimension added")
            #     self.q_table = np.concatenate((self.q_table, np.array([[0,0,0,0,0,0]])))
    elif answer == 'n':
        print("Training a new model!")
        self.logger.info("Creating new q-table.")
        self.q_table = np.zeros((state_space_size, action_space_size)) 

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object i
     passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self.Initial == True:       # Just used for first action in first round, when game_state is None
        self.Initial = False
        self.logger.info(f"Initial=True Loop")
    else:
        #Exploration-exploitation trade-off
        threshold = np.random.random()
        if threshold < self.exploration_rate:
            self.logger.info(f"EXPLORATION")
            self.Explore = True

        # Update Q-table for Q(s,a)
        self.logger.info(f"self_action:{self_action}")
        ind_action = ACTIONS.index(self_action)
        reward = reward_from_events(self,events)
        self.logger.info(f"reward: {reward}")
        self.q_table[get_state_index(self,old_game_state), ind_action] = self.q_table[get_state_index(self,old_game_state), ind_action] * (1 - self.learning_rate) + \
            self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[get_state_index(self,new_game_state), :]))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Exploration rate decay
    if self.exploration_rate > self.min_exploration_rate:
        self.exploration_rate *= self.exploration_decay_rate

    self.Initial = True

    self.logger.info(f"#############\n{np.nonzero(self.q_table)}")
    self.logger.info(self.q_table[np.nonzero(self.q_table)])

    #print(f"EPSILON: {self.exploration_rate}")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        # pickle.dump(self.model, file)
        pickle.dump(self.q_table, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10_000,
        e.GOT_KILLED: -10_000,
        e.SURVIVED_ROUND: 10,
        e.INVALID_ACTION: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.BOMB_DROPPED: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# Function gets number/index of game_state to get access to correct location in q_table
def get_state_index(self, game_state: dict) -> int: 
    self.logger.info(f"## get_state_loop: game_state[coins]: {game_state['coins']}")

    if game_state['coins'] == []:
        return 4 # WAIT
    else: 
        coin_row_index = game_state['coins'][0][0] - 1
        coin_col_index = game_state['coins'][0][1] - 1

        agent_index = 0
        coin_index = 0
        
        for i in range(coin_row_index):
            if i % 2 == 0:
                coin_index += 15
            else:
                coin_index += 8
        
        if (coin_row_index+1) % 2 == 0:
            coin_index += coin_col_index/2
        else: 
            coin_index += coin_col_index

            
        if game_state["coins"][0][0] > game_state["self"][-1][0]:
            coin_index -= 1
        elif game_state["coins"][0][0] == game_state["self"][-1][0] and game_state["coins"][0][1] > game_state["self"][-1][1]:
            coin_index -= 1

        agent_row_index = game_state["self"][-1][0] - 1
        agent_col_index = game_state["self"][-1][1] - 1

        for i in range(agent_row_index):
            if i % 2 == 0:
                agent_index += 15
            else:
                agent_index += 8
        if (agent_row_index+1) % 2 == 0:
            agent_index += agent_col_index/2    
        else: 
            agent_index += agent_col_index
    
        state_index = agent_index * 175 + coin_index

        return np.int(state_index)
