from pynput import keyboard
from rai import RaiEnv
import threading
import pickle
import curses

#stdscr = curses.initscr()
#curses.noecho()

diff = 1
env = RaiEnv(maze_dif=diff)
current_direction = None
pressed_keys = set()
model_reward = []
user_reward = [0, 0, 0, 0, 0, 0]
iteration = 1


for i in range(1, 7):
    with open('models/win_rate_' + str(i) + '.pkl', 'rb') as fp:
        model_reward.append(pickle.load(fp)[-1])

stop_game = False

def move_ball_continuously():
    global pressed_keys, stop_game, model_reward, current_direction
    status = True
    env.view()
    episode_reward = 0
    while status and not stop_game:
        if current_direction is not None:
            next_state, reward, status, win = env.state_update(current_direction)
            episode_reward += reward
    user_reward[diff-1] += episode_reward

    print("User Reward: ", user_reward[diff-1] / iteration, "Model Reward: ", model_reward[diff-1])

# Keyboard event handlers
def on_press(key):
    global current_direction
    if key == keyboard.Key.up:
        current_direction = 1  # up
    elif key == keyboard.Key.left:
        current_direction = 2  # left
    elif key == keyboard.Key.down:
        current_direction = 0  # down
    elif key == keyboard.Key.right:
        current_direction = 3  # right

def on_release(key):
    global stop_game
    if key == keyboard.Key.esc:
        stop_game = True
        return False
    pressed_keys.discard(key)

def start_game():
    global current_direction, stop_game, env, diff, iteration

    # Reset the game variables
    current_direction = None
    stop_game = False

    # Wait for the start key press
    print("Press any key to start the game, or 'ESC' to quit.")
    with keyboard.Events() as events:
        for event in events:
            if isinstance(event, keyboard.Events.Press):
                if event.key == keyboard.Key.esc:
                    return  
                break

    game_thread = threading.Thread(target=move_ball_continuously)
    game_thread.start()
    game_thread.join()  

    del env
    diff += 1
    if(diff == 7):
        diff = 1
        iteration += 1
        
    env = RaiEnv(maze_dif=diff)  

    print(f"Game over! Starting a new game with map difficulty {diff}.")

    start_game()

# ------- MAIN -------
if(__name__ == "__main__"):

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    start_game()

    # Cleanup
    listener.stop()

    curses.echo()
    curses.endwin()