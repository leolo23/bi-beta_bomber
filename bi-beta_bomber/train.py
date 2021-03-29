import time
import events as e
from .my_lib import *
from .my_settings import *


def setup_training(self):
    # Create models folder
    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Create directory and store used settings
    if not os.path.isdir(SETTINGS_DIR):
        os.makedirs(SETTINGS_DIR)
    f = open(f"{SETTINGS_DIR}/{SETTINGS_FILE_NAME}.txt", "w")
    for i in range(len(SETTINGS_INF)):
        f.write(f"{SETTINGS_Variables[i]}: {SETTINGS_INF[i]}\n")
    f.close()

    # parameter for exploration/exploitation in training - will decay
    self.epsilon = EPSILON

    # initialization of parameters for training
    self.state_inputs = None
    self.action = None

    self.current_round_reward = 0
    self.round_rewards = []

    self.inputs_90 = None
    self.inputs_180 = None
    self.inputs_270 = None

    self.position_history = np.zeros((17, 17))
    self.steps_bomb_unused = 0
    self.old_hidden_coins = np.ones((3, 3))


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # FUNCTION IS CALLED BEFORE FIRST STEP!!!!!!!!!!!!
    if old_game_state is None:
        return None
    # print out timestamp and step-round every 10 steps
    if not old_game_state["step"] % 10:
        print(time.strftime("%Y/%m/%d-%H:%M:%S"), "- Round", old_game_state["round"], "- Step", old_game_state["step"])

    # if here round did not end
    done = False

    # update input variables
    old_inputs_90 = self.inputs_90
    old_inputs_180 = self.inputs_180
    old_inputs_270 = self.inputs_270
    self.old_hidden_coins = self.hidden_coins
    new_state_inputs = get_inputs(self, new_game_state)

    # count steps landed on each tile
    self.position_history += map_players([new_game_state["self"]])

    # if bomb not use when possible - increase counter
    if self.prev_bomb_action_state and self.action != 5:
        self.step_bomb_unused += 1
    # if bomb used and was possible (i.e. no invalid action) - reset counter
    elif self.prev_bomb_action_state and self.action == 5:
        self.steps_bomb_unused = 0

    # ADDITIONAL EVENTS and rewards
    events = additional_events(self, old_game_state, self_action, new_game_state, events)
    reward = get_rewards(self, events, old_game_state)
    self.current_round_reward += reward

    # Every step update replay memory with state and rotated states - and train main network
    self.agent.update_replay_memory((self.state_inputs, self.action, reward, new_state_inputs, done))
    self.agent.update_replay_memory((old_inputs_90, rotate_action(90, self.action), reward, self.inputs_90, done))
    self.agent.update_replay_memory((old_inputs_180, rotate_action(180, self.action), reward, self.inputs_180, done))
    self.agent.update_replay_memory((old_inputs_270, rotate_action(270, self.action), reward, self.inputs_270, done))

    self.agent.train(done)

    # updated next step inputs that were already computed here
    self.state_inputs = new_state_inputs


def end_of_round(self, last_game_state, last_action, events):
    # print out time taken to run the current round
    t = round(time.time() - self.t_start_round, 2)
    print(time.strftime("%Y/%m/%d-%H:%M:%S"), "- Round", self.current_round, "done in:", round(t / 60, 2), "min or", t, "s")

    # if here round ended
    done = True

    # Get rewards for last action
    self.old_hidden_coins = self.hidden_coins
    reward = get_rewards(self, events, last_game_state)
    self.current_round_reward += reward

    # Update replay memory for final round and train agent
    self.agent.update_replay_memory((self.state_inputs, self.action, reward, self.state_inputs, done))
    self.agent.update_replay_memory((self.inputs_90, rotate_action(90, self.action), reward, self.inputs_90, done))
    self.agent.update_replay_memory((self.inputs_180, rotate_action(180, self.action), reward, self.inputs_180, done))
    self.agent.update_replay_memory((self.inputs_270, rotate_action(270, self.action), reward, self.inputs_270, done))

    self.agent.train(done)

    # Append episode reward to a list and log stats (every given number of episodes)
    self.round_rewards.append(self.current_round_reward)
    # log on tensorboard and save model
    if not self.current_round % AGGREGATE_STATS_EVERY or self.current_round == 1:
        average_reward = sum(self.round_rewards[-AGGREGATE_STATS_EVERY:]) / len(self.round_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(self.round_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(self.round_rewards[-AGGREGATE_STATS_EVERY:])
        self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                            reward_max=max_reward, epsilon=self.epsilon)

        print("Reward stats: max=", max_reward, "avg=", average_reward, ", min=", min_reward)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            file_name = f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{time.strftime("%Y%m%d-%H%M%S")}.h5'
            print("Saving model:", file_name)
            self.agent.model.save(file_name)

    # decay epsilon
    if self.epsilon > MIN_EPSILON:
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(MIN_EPSILON, self.epsilon)

    print("My score:", last_game_state["self"][1], "\n")


def additional_events(self, old_game_state, self_action, new_game_state, events):
    """
    determine which additional events occurred and append to events list
    """
    if old_game_state is None:
        return events
    # EVENT_NO_CLOSER_TO_COIN --- stage 1 and 2 training
    # check if there are coins available and agent did not collect a coin
    # if len(old_game_state["coins"]) > 0 and e.COIN_COLLECTED not in events:
    #     # get my old and new coordinates
    #     old_my_coord = np.array(old_game_state["self"][3])
    #     new_my_coord = np.array(new_game_state["self"][3])
    #     # get old and new distance from nearest coin in old game state
    #     old_closer_coin_dist = np.min(np.linalg.norm(np.array(old_game_state["coins"]) - old_my_coord, axis=-1))
    #     new_closer_coin_dist = np.min(np.linalg.norm(np.array(old_game_state["coins"]) - new_my_coord, axis=-1))
    #     # if not closer add event
    #     if new_closer_coin_dist >= old_closer_coin_dist:
    #         events.append(EVENT_NO_CLOSER_TO_COIN)

    # EVENT_NOT_FLEEING_OWN_BOMB
    # get my old and new coordinates
    old_my_coord = np.array(old_game_state["self"][3])
    new_my_coord = np.array(new_game_state["self"][3])
    # if bomb was already dropped
    if not old_game_state["self"][2]:
        # if on explosion tile of own bomb - state_inputs[169] is center tile of local own bomb map
        if self.state_inputs[0][169] > 0:
            # get old and new distance from dropping point of my bomb
            my_bomb_coord = np.array(self.my_bomb_coord)
            old_dist_bomb = np.linalg.norm(old_my_coord - my_bomb_coord)
            new_dist_bomb = np.linalg.norm(new_my_coord - my_bomb_coord)
            # if not farther away from dropping point add event
            if new_dist_bomb <= old_dist_bomb:
                events.append(EVENT_NOT_FLEEING_OWN_BOMB)

    # EVENT_NO_CLOSER_TO_NEAR_COIN
    closer_to_near_coin = False
    # check if there are available coins
    if len(old_game_state["coins"]) > 0:
        # get nearest coin old and new distance
        old_coin_min_dist = np.min(np.linalg.norm(old_my_coord - np.array(old_game_state["coins"]), axis=0))
        old_coin_new_dist = np.min(np.linalg.norm(new_my_coord - np.array(old_game_state["coins"]), axis=0))
        # if the nearest coin was in the field of view, i.e. less than 3 tiles away
        if old_coin_min_dist < 3:
            # if not closer to the coin add event
            if old_coin_new_dist >= old_coin_min_dist:
                events.append(EVENT_NO_CLOSER_TO_NEAR_COIN)
            # else set the variable accordingly for the looping check
            else:
                closer_to_near_coin = True

    # EVENT_NOT_USING_BOMB
    # if did not use bomb for too long and not getting closer to a near coin
    if self.steps_bomb_unused > 3 and not closer_to_near_coin:
        events.append(EVENT_NOT_USING_BOMB)

    # EVENT_NO_CLOSER_TO_AVAILABLE_COIN
    # get pooled maps
    me_pooled = my_pooling_2d(map_players([old_game_state["self"]])[1:-1, 1:-1], 5, 5)
    coins_pooled = my_pooling_2d(map_coins(old_game_state["coins"])[1:-1, 1:-1], 5, 5)
    # if not closer to coin in FOV, there are available coins and agent is not on a region with available nor hidden coins
    # track if event could occur
    check_dist_a_coin = False
    # if not getting closer to coin in FOV, there are available coins, and agent is not on window with an available or hidden coin
    if not closer_to_near_coin and np.sum(coins_pooled) > 0 and not np.sum(me_pooled * coins_pooled) and not np.sum(
            me_pooled * self.old_hidden_coins):
        # get old and new distance
        old_dist_a_coin = np.linalg.norm(old_my_coord - POOLED_CENTRE_COORD, axis=-1) * coins_pooled.flatten()
        old_dist_a_coin = np.min(old_dist_a_coin[old_dist_a_coin > 0])  # if here there it exists
        new_dist_a_coin = np.linalg.norm(new_my_coord - POOLED_CENTRE_COORD, axis=-1) * coins_pooled.flatten()
        if len(new_dist_a_coin[new_dist_a_coin > 0]):
            new_dist_a_coin = np.min(new_dist_a_coin[new_dist_a_coin > 0])
        else:
            new_dist_a_coin = 0
        check_dist_a_coin = True

    # EVENT_NO_CLOSER_TO_HIDDEN_COIN
    # if not getting closer to coin in FOV, there are hidden coins, and agent is not on window with an available or hidden coin
    # track if event could occur
    check_dist_h_coin = False
    if not closer_to_near_coin and np.sum(self.old_hidden_coins) and not np.sum(
            me_pooled * self.old_hidden_coins) and np.sum(me_pooled * coins_pooled):
        # get old and new distance
        old_dist_h_coin = np.linalg.norm(old_my_coord - POOLED_CENTRE_COORD, axis=-1) * self.old_hidden_coins.flatten()
        old_dist_h_coin = np.min(old_dist_h_coin[old_dist_h_coin > 0])  # if here there it exists
        new_dist_h_coin = np.linalg.norm(new_my_coord - POOLED_CENTRE_COORD, axis=-1) * self.old_hidden_coins.flatten()
        if len(new_dist_h_coin[new_dist_h_coin > 0]):
            new_dist_h_coin = np.min(new_dist_h_coin[new_dist_h_coin > 0])
        else:
            new_dist_h_coin = 0
        check_dist_h_coin = True

    # determine which occurred - in proximity prefer available coin
    if check_dist_a_coin and check_dist_h_coin:
        if old_dist_a_coin < 10:
            if new_dist_a_coin >= old_dist_a_coin:
                events.append(EVENT_NO_CLOSER_TO_AVAILABLE_COIN)
            else:
                closer_to_near_coin = True
        else:
            if new_dist_h_coin >= old_dist_h_coin:
                events.append(EVENT_NO_CLOSER_TO_HIDDEN_COIN)
            else:
                closer_to_near_coin = True
    elif check_dist_a_coin:
        if new_dist_a_coin >= old_dist_a_coin:
            events.append(EVENT_NO_CLOSER_TO_AVAILABLE_COIN)
        else:
            closer_to_near_coin = True
    elif check_dist_h_coin:
        if new_dist_h_coin >= old_dist_h_coin:
            events.append(EVENT_NO_CLOSER_TO_HIDDEN_COIN)
        else:
            closer_to_near_coin = True

    # EVENT_NO_CLOSER_TO_ENEMIES
    # if there are no more coins at all look for enemies
    # get pooled maps
    others = map_players(old_game_state["others"])
    others_pooled = my_pooling_2d(others[1:-1, 1:-1], 5, 5)
    l_others = get_local_map(self, old_game_state["self"][3], others)
    # if there are no more coins and there enemies but not in FOV
    if not np.sum(coins_pooled) and not np.sum(self.old_hidden_coins) and np.sum(others_pooled) and not np.sum(
            me_pooled * others_pooled):
        # get old and new distance
        old_dist_enemy = np.linalg.norm(old_my_coord - POOLED_CENTRE_COORD, axis=-1) * others_pooled.flatten()
        old_dist_enemy = np.min(old_dist_enemy[old_dist_enemy > 0])  # if here there are still enemies
        new_dist_enemy = np.linalg.norm(new_my_coord - POOLED_CENTRE_COORD, axis=-1) * others_pooled.flatten()
        if len(new_dist_enemy[new_dist_enemy > 0]):
            new_dist_enemy = np.min(new_dist_enemy[new_dist_enemy > 0])
        else:
            new_dist_enemy = 0
        # if not closer add event - else keep track of useful action not to be penalized by loop check
        if new_dist_enemy >= old_dist_enemy:
            events.append(EVENT_NO_CLOSER_TO_ENEMIES)
        else:
            # to use against step looping check
            closer_to_near_coin = True

    # EVENT_STEP_LOOPING
    # get my new x and y coordinated
    my_new_x = new_game_state["self"][3][0]
    my_new_y = new_game_state["self"][3][1]
    # if stepping again on tile and was not fleeing from bomb (has bomb action) and did not use bomb and not chasing a near coin
    if self.position_history[my_new_x, my_new_y] > 2 and old_game_state["self"][2] and self.action != 5 and not closer_to_near_coin:
        events.append(EVENT_STEP_LOOPING)

    return events


def get_rewards(self, events, gs):
    """
    assign rewards for happened events
    """
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    if e.SURVIVED_ROUND in events:
        print("Survived round")
    #     reward_sum += 400 - gs["step"]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
