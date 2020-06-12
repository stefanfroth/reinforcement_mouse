import argparse
import time

import cv2

from firstvisitmontecarlo import FirstVisitStateValueMonteCarloMouse
from grid import Grid
from mouse import Mouse
from temporaldifference import QLearningMouse, Sarsa

## Set global variables ##
# Sleeping time
SLEEP = 0.5
EPISODES = 1


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        description="A mouse learning to get to the cheese using reinforcement learning"
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="Mouse",
        help="""Choose the learning algorithm. Available are
                        Mouse, FVSVMC, Sarsa and QLearning""",
    )
    parser.add_argument(
        "-e",
        "--exploration",
        action="store_true",
        help="""Determines whether the mouse will explore or only
                        exploit.""",
    )
    parser.add_argument(
        "-d",
        "--draws",
        type=int,
        default=5,
        help="Nr. of draws before updating the policy.",
    )

    args = parser.parse_args()

    if args.method == "Mouse":
        mouse = Mouse()
    elif args.method == "FVSVMC":
        mouse = FirstVisitStateValueMonteCarloMouse(nr_of_sample_draws=args.draws)
    elif args.method == "Sarsa":
        mouse = Sarsa(exploration=args.exploration)
    elif args.method == "QLearning":
        mouse = QLearningMouse(exploration=args.exploration)
    grid = Grid(mouse)

    while True:

        # Print the mouses rewards
        # print(f'''The mouses rewards are:\n
        #           Timestep: {mouse.reward_timestep}\n
        #           Episode:  {mouse.reward_episode}\n
        #           Total: {mouse.reward_total}\n''')

        # Reset the mouse and increase the episode
        if mouse.state in [1, 9]:
            grid.draw()

            cv2.imshow("frame", grid.frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
            # time.sleep(SLEEP)
            if cv2.waitKey(0) == ord("a"):
                continue

            # mouse.update_state_values(terminal=True)
            # mouse.update_action_values()
            mouse.transition_to_new_episode()

        # Q-values
        # print(f'The Q-values of the mouse are \n {mouse.q}')
        # print(f'The mouse is currently in state {mouse.state}')

        grid.draw()

        cv2.imshow("frame", grid.frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        mouse.play_one_round()
        print(f"The mouse chooses action {mouse.action}")
        # time.sleep(SLEEP)
        if cv2.waitKey(0) == ord("a"):
            continue

    cv2.destroyAllWindows()
