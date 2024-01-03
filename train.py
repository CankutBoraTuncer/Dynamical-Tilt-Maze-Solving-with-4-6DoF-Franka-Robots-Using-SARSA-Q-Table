import pickle
import random
import numpy as np
from rai import RaiEnv
import matplotlib.pyplot as plt

def train(env):
    global start_list, win_history, cumulative_reward_history, win_rate_dict, win_rate_history, reward_history
    global discount, learning_rate, eligibility_decay, exploration_decay, cumulative_reward, exploration_rate
    global exploitation_decay, margin
    global win_count, shortest_path_rate, episode
    
    if len(start_list) == 0:
        start_list = env.empty_points.copy()
        
    state = tuple(env.init_position)
    action = random.choice(list(env.action_map.keys()))

    episode_reward = 0
    etrace = dict()
    _ = env.reset(env.init_position)
    
    margin = margin * exploitation_decay
    min_reward = len(env.reward_path) * 0.25 * margin

    status = True
    while status:
        try:
            etrace[(state, action)] += 1
        except KeyError:
            etrace[(state, action)] = 1

        next_state, reward, status, win = env.state_update(action)
        win_count += win
        next_state = tuple(next_state)
        next_action = env.predict(tuple(next_state))

        if np.random.random() < exploration_rate:
            next_action = random.choice(list(env.action_map.keys()))
        else:
            next_action = env.predict(tuple(next_state))

        cumulative_reward += reward
        episode_reward += reward
        print("Episode Reward: ", episode_reward, "Min Reward:", -min_reward)
        if episode_reward < -min_reward:
            status = False

        if (state, action) not in env.Q.keys():
            env.Q[(state, action)] = 0.0

        next_Q = env.Q.get((next_state, next_action), 0.0)

        delta = reward + discount * next_Q - env.Q[(state, action)]

        for key in etrace.keys():
            env.Q[key] += learning_rate * delta * etrace[key]

        # decay the eligibility trace
        for key in etrace.keys():
            etrace[key] *= (discount * eligibility_decay)

        state = next_state
        action = next_action
        
    print('Win Rate: ', win_count/episode)
    win_rate_history.append(win_count/episode)
    reward_history.append(episode_reward)
    cumulative_reward_history.append(cumulative_reward)
    exploration_rate = exploration_rate * exploration_decay

    print("Status:" , win)
    episode_reward = 0

# ------- MAIN -------

if(__name__ == "__main__"):
    difficulty = 7
    r = random.randint(0,1000)
    env = RaiEnv(difficulty)
    random.seed(42)
    discount = 0.9
    learning_rate = 0.10
    eligibility_decay = 0.80
    exploration_decay = 0.999
    exploitation_decay = 0.999
    margin = 10
    cumulative_reward = 0
    cumulative_reward_history = []
    reward_history = []
    shortest_path_rate = []
    win_history = []
    exploration_rate = 0.10 
    win_rate_dict = dict()
    win_rate_history = []
    start_list = env.empty_points.copy()
    win_count = 0
    num_epochs = 5000
    
    for episode in range(1, num_epochs):
        print("Episode: ", episode)
        train(env)
        if(episode % 50 == 0):
            with open('models/q_table_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) + '.pkl', 'wb') as fp:
                pickle.dump(env.Q, fp)
            with open('models/cum_reward_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +    '.pkl', 'wb') as fp:
                pickle.dump(cumulative_reward_history, fp)
            with open('models/reward_hist_last_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +   '.pkl', 'wb') as fp:
                pickle.dump(reward_history, fp)
            with open('models/win_rate_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +   '.pkl', 'wb') as fp:
                pickle.dump(win_rate_history, fp)

    with open('models/q_table_last_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +  '.pkl', 'wb') as fp:
        pickle.dump(env.Q, fp)
    with open('models/cum_reward_last_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +   '.pkl', 'wb') as fp:
        pickle.dump(cumulative_reward_history, fp)
    with open('models/reward_hist_last_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +    '.pkl', 'wb') as fp:
        pickle.dump(reward_history, fp)
    with open('models/win_rate_last_' + str(difficulty) + '_' + str(round(win_count/episode, 5)) + "_pos_vel_end" + str(episode) +   '.pkl', 'wb') as fp:
        pickle.dump(win_rate_history, fp)

    # Create a figure
    fig = plt.figure()

    # Add three subplots
    # The arguments to add_subplot are (rows, columns, index)
    ax1 = fig.add_subplot(3, 1, 1) # First subplot
    ax2 = fig.add_subplot(3, 1, 2) # Second subplot
    ax3 = fig.add_subplot(3, 1, 3) # Third subplot

    # Plot data on each subplot
    ax1.plot(win_rate_history)
    ax2.plot(cumulative_reward_history)
    ax3.plot(reward_history)

    plt.show()




