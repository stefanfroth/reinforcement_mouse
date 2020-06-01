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
    Defines the methods and attributes of the mouse in the game.

    Parameters:
    -----------
    states:            states the mouse can be in, depends on the grid
    starting_position: in which position does the mouse start
    exploration:       True means the mouse is exploring, False means it is not
    discount_factor:   the discount factor of the game
    '''

    def __init__(self, states=list(range(1, 10)), starting_position=5,
                 exploration=False, discount_factor=0.8):

        # All possible states
        self.states = states

        # State in timesteps t and t-1
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

        # Rewards in timesteps t and t-1
        self.reward = 0
        self.last_reward = 0

        # Policy: Choose an action given the state; initially random
        self.policy = pd.DataFrame(np.ones(shape=(9,4)) * 0.25,
                                   columns=['up', 'right', 'left', 'down'],
                                   index=self.states)

        # Episode; start with 1
        self.episode = 1

        # Wether to explore or just to exploit
        self.exploration = exploration

        # Discount factor
        self.discount_factor = discount_factor

        # Timestep t
        self.timestep

        ### Cutoff: The rest of the __init__ function belongs to subclasses

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

        # Save the visited states
        self.state_history = defaultdict(list)


    def choose_action(self):
        '''
        Determines the next state of the mouse.
        '''
        self.last_state = self.state

        # Implement exploration/exploitation
        if self.exploration:
            epsilon = np.random.randint(1, 100)/(100+self.episode)
        else:
            epsilon = 0.1

        '''With probability epsilon, choose randomly
           else according to the policy'''
        if epsilon > 0.9 or self.episode < 4:
            self.action = np.random.choice(self.actions,
                                          p=self.policy.loc[self.state].values)
        else:
            # Here is immense potential for refactoring
            best_action = []
            value_best = -1000
            for action in self.actions:
                action_value = self.state_value\
                               [self.transition[self.state][action]]
                if action_value > value_best:
                    best_action = [action]
                    value_best = action_value
                elif action_value == value_best:
                    best_action.append(action)
            print(f'The actions are {self.actions}')
            print(f'The state_values are {[self.state_value[self.transition[self.state][action]] for action in self.actions]}')
            print(f'The best possible actions are {best_action}')

            # If the best_action is non-unique
            self.action = np.random.choice(best_action)

        # Save the new state
        self.state = self.transition[self.last_state][self.action]

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
        self.reward += self.reward_function[self.state]
        self.last_reward = self.reward_function[self.state]
        # if self.state == 1:
        #     self.last_reward = 1
        #     self.reward += 1

            # Increase the state_reward
        #    self.increase_state_reward_occurence(self.reward)

        # elif self.state == 9:
        #     self.last_reward = 2
        #     self.reward += 2

            # Increase the state reward
        #    self.increase_state_reward_occurence(self.reward)

        # else:
        #     self.last_reward = -1
        #     self.reward -= 1

        if self.state in [1, 9]:
            self.increase_state_reward_occurence(self.reward)

    ### The following methods are only needed for the subclasses

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
        ## The combined list is not used, therefore commented out for now
        # combined_list = []
        # for values in self.state_history.values():
        #     combined_list.extend(set(values))

        # Update (v_{t-1} + alpha * (v_t - v_{t-1}))
        for state in self.states:
            if self.state_occurence[state] > 0:
                average_reward = self.state_reward[state]/self.state_occurence[state]

                # Distinction betweent positive and negative values
                self.state_value[state] = self.state_value[state] + self.learning_rate * \
                                          (average_reward - self.state_value[state])

    def reset_state_occurence(self):
        '''
        Resets the state occurence of the first policy
        '''
        ...

    '''First-visit MC prediction, for estimating V'''


    class PolicyIterationMouse(Mouse):
        ...

    class ValueIterationMouse(Mouse):
        ...

    class MonteCarloMouse(Mouse):
        ...

    class TemporalDifferenceMouse(Mouse):
        ...
