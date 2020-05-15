'''
The module mouse defines a mouse that learns to eat cheese via reinforcement
learning.
'''

## Next to-dos: - Implement updated action,
#               - Implement discount factor


import pandas as pd
import numpy as np
from collections import defaultdict


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
        self.actions = ['up', 'right', 'left', 'down']
        # self.action_mapping = {
        #                         0: 'up',
        #                         1: 'right',
        #                         2: 'left',
        #                         3: 'down'
        #                         }

        # Save last action
        self.action = ''

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

        # Create helper function for the calculation of the state-values
        # It is called state reward
        self.state_reward = {1: 0,
                             2: 0,
                             3: 0,
                             4: 0,
                             5: 0,
                             6: 0,
                             7: 0,
                             8: 0,
                             9: 0,
                            }

        # In how many episodes does the state occur?
        self.state_occurence = {1: 0,
                                2: 0,
                                3: 0,
                                4: 0,
                                5: 0,
                                6: 0,
                                7: 0,
                                8: 0,
                                9: 0,
                                }

        # Create the state-value function (first as a dictionary, for rendering
        # as a matrix)
        self.state_value = {1: 0,
                            2: 0,
                            3: 0,
                            4: 0,
                            5: 0,
                            6: 0,
                            7: 0,
                            8: 0,
                            9: 0,
                            }

        # Learning Rate
        self.learning_rate = 0.5

        # Episode; start with 1
        self.episode = 1

        # Save the visited states
        self.state_history = defaultdict(list)


    # def find_best_state(self):


    def choose_action(self):
        '''
        Determines the next state of the mouse.
        '''
        self.last_state = self.state

        # Implement exploration/exploitation
        # epsilon = np.random.randint(1, 100)/100+self.episode
        epsilon = 0.1

        '''With probability epsilon, choose randomly, else the best'''
        if epsilon > 0.9 or self.episode < 4:
            self.action = np.random.choice(self.actions,
                                          p=self.policy.loc[self.state].values)
        else:
            best_action = []
            value_best = -1000
            for action in self.actions:
                action_value = self.state_value[self.transition.loc[self.state, action]]
                if action_value > value_best:
                    best_action = [action]
                    value_best = action_value
                elif action_value == value_best:
                    best_action.append(action)
            print(f'The actions are {self.actions}')
            print(f'The state_values are {[self.state_value[self.transition.loc[self.state, action]] for action in self.actions]}')
            print(f'The best possible actions are {best_action}')

            # If the best_action is unique
            self.action = np.random.choice(best_action)

        # Save the new state
        self.state = self.transition.loc[self.last_state, self.action]

        # Save the new state in the state history
        self.state_history[self.episode].append(self.state)

    def increase_state_reward_occurence(self, reward):
        '''
        Increases the state_reward used to calculate the state_value.
        '''
        for state in self.states:
            if state in self.state_history[self.episode]:
                self.state_reward[state] += reward
                self.state_occurence[state] += 1

    def assign_reward(self):
        '''
        Assigns a reward to the mouse given its chosen action.
        '''
        if self.state == 1:
            self.last_reward = 1
            self.reward += 1

            # Increase the state_reward
            self.increase_state_reward_occurence(self.reward)

        elif self.state == 9:
            self.last_reward = 2
            self.reward += 2

            # Increase the state reward
            self.increase_state_reward_occurence(self.reward)

        else:
            self.last_reward = -1
            self.reward -= 1

    def update_q(self):
        '''
        Updates the action_value (Q-)function of the mouse.
        '''
        self.q.loc[self.last_state, self.action] = \
            self.q.loc[self.last_state, self.action] + \
            self.learning_rate * (self.last_reward - \
            self.q.loc[self.last_state, self.action]) +\
            sum(self.q.loc[self.state])/4

    def update_state_values(self):
        '''
        Updates the values of the state_value function.
        '''
        # This code is specifically written for a Monte Carlo implementation
        combined_list = []
        for values in self.state_history.values():
            combined_list.extend(set(values))

        # Update (v_{t-1} + alpha * (v_t - v_{t-1}))
        for state in self.states:
            if self.state_occurence[state] > 0:
                average_reward = self.state_reward[state]/self.state_occurence[state]

                # Distinction betweent positive and negative values
                self.state_value[state] = self.state_value[state] + self.learning_rate * \
                                          (average_reward - self.state_value[state])
