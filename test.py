import pickle
from rai import RaiEnv
import time

if(__name__ == "__main__"):

    difficulty = 4
    env = RaiEnv(difficulty)
    env.view()

    with open('models/q_table_' + str(difficulty) + '.pkl', 'rb') as fp:
        env.Q = pickle.load(fp)

    for i in range(1, 20):
        print("Iteration No: ", i)
        state = env.reset(env.init_position)
        exit_counter = 0

        while env.over == False:
            if exit_counter > 1000:
                break
            action = env.predict(tuple(state))
            state, reward, status, win = env.state_update(action)
            exit_counter += 1
            
            
