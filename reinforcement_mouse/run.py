import cv2
import time
from mouse import Mouse, TemporalDifferenceMouseStateValues, Sarsa
from firstvisitmontecarlo import FirstVisitStateValueMonteCarloMouse
from grid import Grid

## Set global variables ##
# Sleeping time
SLEEP = 0.5
EPISODES = 5

# mouse = FirstVisitStateValueMonteCarloMouse(nr_of_sample_draws=5)
# mouse = TemporalDifferenceMouseStateValues()
mouse = Sarsa(exploration=True)
grid = Grid(mouse)

while True:

    # Print the mouses rewards
    print(f'''The mouses rewards are:\n
              Timestep: {mouse.reward_timestep}\n
              Episode:  {mouse.reward_episode}\n
              Total: {mouse.reward_total}\n''')

    # Reset the mouse and increase the episode
    if mouse.state in [1, 9]:
        grid.draw()

        cv2.imshow('frame', grid.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(SLEEP)

        # mouse.update_state_values(terminal=True)
        mouse.update_action_values()
        mouse.transition_to_new_episode()


    # Q-values
    # print(f'The Q-values of the mouse are \n {mouse.q}')
    # print(f'The mouse is currently in state {mouse.state}')


    grid.draw()

    cv2.imshow('frame', grid.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    mouse.play_one_round()
    print(f'The mouse chooses action {mouse.action}')
    time.sleep(SLEEP)

    # If the number of episodes to train are over, calculate the new
    # state_values
    # if mouse.episode % EPISODES == 0 and mouse.state in [1, 9]:
    #     mouse.update_state_values()
    #     mouse.reward = 0
    #     print(f"The mouse's state values are {mouse.state_value}")
    #     time.sleep(3)


cv2.destroyAllWindows()
