'''Defines a mouse that walks through a grid and searches for cheese.'''

## Next to-dos: - Implement updated action,
#               - Implement discount factor


import pandas as pd
import numpy as np
# from collections import defaultdict


class Mouse:
    '''
    Defines the methods and attributes of the mouse in the game.

    Parameters:
    -----------
    states:            States the mouse can be in, depends on the grid
    starting_position: In which position does the mouse start
    exploration:       True means the mouse is exploring, False means it is not
    discount_factor:   The discount factor of the game
    learning_rate:     Learning rate at which the value is updated
    '''

    # Take exploration out of the simple mouse
    def __init__(self, states=list(range(1, 10)), starting_position=5,
                 exploration=False, discount_factor=0.8, learning_rate=0.5):

        # All possible states
        self.states = states

        # State in timesteps t and t-1
        self.starting_position = starting_position
        self.state = starting_position
        self.last_state = starting_position

        # All possible actions
        self.actions = ['up', 'right', 'left', 'down']

        # Action in timestep t
        self.action = ''

        # State Transition Function: Mapping between actions and states
        ''' Keys of dictionary {state: {up: next_state,
                                        right: next_state, ...}}'''
        self.transition = {
                            1: {'up': 7, 'right': 2, 'left': 3, 'down': 4},
                            2: {'up': 8, 'right': 3, 'left': 1, 'down': 5},
                            3: {'up': 9, 'right': 1, 'left': 2, 'down': 6},
                            4: {'up': 1, 'right': 5, 'left': 6, 'down': 7},
                            5: {'up': 2, 'right': 6, 'left': 4, 'down': 8},
                            6: {'up': 3, 'right': 4, 'left': 5, 'down': 9},
                            7: {'up': 4, 'right': 8, 'left': 9, 'down': 1},
                            8: {'up': 5, 'right': 9, 'left': 7, 'down': 2},
                            9: {'up': 6, 'right': 7, 'left': 8, 'down': 3}
                            }

        # Reward Function {state: reward}
        # Should this be an attribute of the grid in comparison with the states
        self.reward_function = {
                                1: 1,
                                2: 0,
                                3: 0,
                                4: 0,
                                5: 0,
                                6: 0,
                                7: 0,
                                8: 0,
                                9: 2
                                }

        # Rewards in for timestep t, for the episode and for the whole history
        self.reward_timestep = 0
        self.reward_episode = 0
        self.reward_total = 0

        # Policy: Choose an action given the state; initially random
        self.policy = pd.DataFrame(np.ones(shape=(9,4)) * 0.25,
                                   columns=['up', 'right', 'left', 'down'],
                                   index=self.states)

        # Episode; start with 1
        self.episode = 1

        # Discount factor
        self.discount_factor = discount_factor

        # Timestep t
        self.timestep = 0

        # Wether to explore or just to exploit
        self.exploration = exploration

        # Learning Rate
        self.learning_rate = learning_rate


    def choose_action(self):
        '''
        Determines the action the mouse takes
        '''
        self.action = np.random.choice(self.actions, p=policy.iloc[self.state])


    def transition_to_next_state(self):
        '''
        Transitions to the next state conditional on the transition function
        and the chosen action
        '''
        self.last_state = self.state
        self.state = self.transition[self.state][self.action]


    def assign_reward(self):
        '''
        Assigns a reward to the mouse given its chosen action.
        '''
        self.reward_timestep = self.reward_function[self.state]

        self.reward_episode += self.reward_timestep * \
                               self.discount_factor**self.timestep
        self.reward_total += self.reward_timestep

        # increase timestep
        self.timestep += 1

    ## Perspectively should be moved to the grid
    def transition_to_new_episode(self):
        '''
        After the mouse finds the cheese the episode is over and a new episode
        starts
        '''
        # Move back to the starting_position
        self.state = self.starting_position
        self.last_state = self.starting_position

        # Increase the total reward and reset reward_episode and reward_timestep
        self.reward_timestep = 0
        self.reward_episode = 0

        # Set the number of episodes one up
        self.episode += 1

        # Reset timestep
        self.timestep = 0

    def play_one_round(self):
        self.choose_action()
        self.transition_to_next_state()
        self.assign_reward()


class PolicyIterationMouse(Mouse):
    ...

class ValueIterationMouse(Mouse):
    ...

class OnPolicyFirstVisitMCControl(Mouse):
    '''
    P. 101/123 book
    '''
    ...
