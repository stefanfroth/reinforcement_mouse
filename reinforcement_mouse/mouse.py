'''
The module mouse defines a mouse that learns to eat cheese via reinforcement
learning.
'''

import pandas as pd
import numpy as np
import time


class Mouse:
    '''
    The class Mouse defines the data and attributes of the mouse in the game.
    '''

    def __init__(self, states=list(range(1, 10)),
                 starting_position=5):
        # Reward
        self.reward = 0

        # Reward in the last period
        self.last_reward = 0

        # The starting state of the grid
        self.state = starting_position
        self.last_state = starting_position

        # All possible states
        self.states = states

        # All possible actions
        self.actions = [0, 1, 2, 3] #up, right, down, left
        self.action_mapping = {
                                0: 'up',
                                1: 'right',
                                2: 'left',
                                3: 'down'
                                }

        # Save last action
        self.action = 0

        # Mapping between actions and states
        self.transition = pd.DataFrame([
                                     [7, 2, 3, 4],
                                     [8, 3, 1, 5],
                                     [9, 1, 2, 6],
                                     [1, 5, 6, 7],
                                     [2, 6, 4, 8],
                                     [3, 4, 5, 9],
                                     [4, 8, 9, 1],
                                     [5, 9, 7, 2],
                                     [6, 7, 8, 3]
                                    ],
                                    index=self.states,
                                    columns=['up', 'right', 'left', 'down'])

        # Choose an action given the state
        self.policy = pd.DataFrame(np.ones(shape=(9,4)) * 0.25,
                                   columns=['up', 'right', 'left', 'down'],
                                   index=self.states)

        # Create the action-value function
        self.q = pd.DataFrame(np.zeros(shape=(9,4)),
                              columns=['up', 'right', 'left', 'down'],
                              index=self.states)

        # Create the state-value function
        self.sv = pd.DataFrame(np.zeros(3,3))

        # Learning Rate
        self.learning_rate = 0.5


    def find_best_state(self):


    def choose_action(self):
        '''
        The method chose_action determines the next state of the mouse.
        '''
        self.last_state = self.state
        self.action = np.random.choice(self.actions,
                                      p=self.policy.loc[self.state].values)
        self.state = self.transition.loc[self.last_state, self.action_mapping[self.action]]

    def assign_reward(self):
        '''
        The method assign_reward assigns a reward to the mouse given the action
        it chose.
        '''
        if self.state == 1:
            self.last_reward = 1
            self.reward += 1
        elif self.state == 9:
            self.last_reward = 2
            self.reward += 2
        else:
            self.last_reward = 0

    def update_values(self):
        '''
        The method update_values updates the value function of the mouse.
        '''
        self.q.loc[self.last_state, self.action_mapping[self.action]] = \
            self.q.loc[self.last_state, self.action_mapping[self.action]] + \
            self.learning_rate * (self.last_reward - \
            self.q.loc[self.last_state, self.action_mapping[self.action]]) +\
            sum(self.q.loc[self.state])/4
