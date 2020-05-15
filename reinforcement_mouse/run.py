import cv2
import time
from mouse import Mouse
from grid import Grid

## Set global variables ##
# Sleeping time
SLEEP = 1
EPISODES = 5

mouse = Mouse()
grid = Grid(mouse)

while True:

    # Reset the mouse and increase the episode
    if mouse.state in [1, 9]:
        grid.draw()

        cv2.imshow('frame', grid.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(SLEEP)

        mouse.state = 5
        mouse.last_state = 5
        mouse.last_reward = 0
        mouse.episode += 1


    # Q-values
    # print(f'The Q-values of the mouse are \n {mouse.q}')
    # print(f'The mouse is currently in state {mouse.state}')


    grid.draw()

    cv2.imshow('frame', grid.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    mouse.choose_action()
    mouse.assign_reward()
    print(f'The mouse chooses action {mouse.action}')
    time.sleep(SLEEP)

    # If the number of episodes to train are over, calculate the new
    # state_values
    if mouse.episode % EPISODES == 0 and mouse.state in [1, 9]:
        mouse.update_state_values()
        mouse.reward = 0
        print(f"The mouse's state values are {mouse.state_value}")
        time.sleep(3)


cv2.destroyAllWindows()
