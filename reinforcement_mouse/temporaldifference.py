'''
Module containing different implementations of temporal difference learning
algorithms.
'''

import pandas as pd
import numpy as np
from mouse import Mouse

class Sarsa(Mouse):
    '''Implementation of Sarsa solution'''

    def __init__(self, states=list(range(1, 10)), starting_position=5,
                 exploration=False, discount_factor=0.8,
                 learning_rate=0.5):
        '''
        Constructor of Sarsa.

        Additionally to the parents constructor I need action values and
        a method to update action values.
        '''
        super().__init__(states, starting_position, exploration,
                         discount_factor, learning_rate)

        # Create the action-value function
        self.q = pd.DataFrame(np.zeros(shape=(9,4)),
                              columns=['up', 'right', 'left', 'down'],
                              index=self.states)

    def print_exploration(self):
        '''Debugging'''
        print(f'The value of exploration is {self.exploration}')

    def choose_action(self):
        '''
        Extends the choose_action method of the parent class
        '''
        ## Given that this choose_action method is basically the same as for the
        ## FirstVisitStateValueMonteCarloMouse this should be part of the Mouse
        # Implement exploration/exploitation
        if not self.exploration:
            self.action = np.random.choice(self.actions, p=self.policy.loc[self.state])
        else:
            epsilon = np.random.randint(1, 100)/(100 + self.episode)
            print(f'I am exploring if epsilon is > 0.5 and it is {epsilon}')

            '''With probability epsilon, choose randomly
               else according to the policy'''
               ## Make the episode variable
            if epsilon > 0.5:# or self.episode < 4:
                self.action = np.random.choice(self.actions)
            else:
                self.action = np.random.choice(self.actions, p=self.policy.loc[self.state])

        print(f'''The action_values are \n
                {self.q}''')


    def update_action_values(self, terminal=False):
        '''
        Update the action values
        '''
        # The action value of the last state and the action taken
        value_s_0 = self.q.at[self.last_state, self.action]

        # The discounted action values of the new state given the policy
        prob = 0
        value = 0
        for action in self.actions:
            prob = self.policy.loc[self.state, action]
            value += prob*self.q.loc[self.state, action]

        discounted_value_s_1 = self.discount_factor * value

        # The update applied to the state value
        update = self.learning_rate * \
                 (self.reward_timestep + discounted_value_s_1 - value_s_0)

        # The new state value
        self.q.loc[self.last_state, self.action] = value_s_0 + update


    # Can probably done a lot easier
    def update_policy(self):
        '''
        Update the policy to choose the optimal action
        '''
        for state in self.states:
            best_actions = []
            best_value = 0
            for action in self.actions:
                value = self.q.loc[state, action]
                if  value > best_value:
                    best_actions = [action]
                    best_value = value
                elif value == best_value:
                    best_actions.append(action)

            self.policy.loc[state] = [1/len(best_actions) if x in best_actions else 0 for x in self.actions]


    def play_one_round(self):
        '''Extended method of mouse'''
        super().play_one_round()
        self.update_action_values()
        self.update_policy()
        self.print_exploration()


# Implement the Q-Learning Mouse
class QLearningMouse(Mouse):
    '''Implement Q-Learning algorithm for the Mouse'''
    ...

## This Mouse is inherently flawed as using state values for temporal difference
## learning is not possible?
class TemporalDifferenceMouseStateValues(Mouse):
    '''
    Implements temporal difference learning for the state value function.

    "Tabular TD(0) for estimating v⇡" P. 120/142 book

    Parameters:
    -----------
    states:            States the mouse can be in, depends on the grid
    starting_position: In which position does the mouse start
    exploration:       True means the mouse is exploring, False means it is not
    discount_factor:   The discount factor of the game
    learning_rate:     Learning rate at which the value is updated
    '''
    ## The __init__ is exactly the same as for the FirstVisitStateValueMonteCarloMouse.
    ## Should be a StateVale Mouse
    def __init__(self):
        super().__init__(states=list(range(1, 10)), starting_position=5,
                     exploration=False, discount_factor=0.8,
                     learning_rate=0.5)

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


    def choose_action(self):
        '''
        Extends the choose_action method of the parent class
        '''
        ## Given that this choose_action method is basically the same as for the
        ## FirstVisitStateValueMonteCarloMouse this should be part of the Mouse
        # Implement exploration/exploitation
        if not self.exploration:
            self.action = np.random.choice(self.actions, p=self.policy.loc[self.state])
        else:
            epsilon = np.random.randint(1, 100)/(100+self.episode)

            '''With probability epsilon, choose randomly
               else according to the policy'''
               ## Make the episode variable
            if epsilon > 0.9 or self.episode < 4:
                self.action = np.random.choice(self.actions)

        print(f'The state_values are {self.state_value}')


    # Transitioning is just inherited from Mouse
    # def transition_to_next_state(self):


    def update_state_values(self, terminal=False):
        '''
        Update the state values
        '''
        if not terminal:
            # The value of the last state
            value_s_0 = self.state_value[self.last_state]

            # The discounted value of the new state
            discounted_value_s_1 = self.discount_factor * self.state_value[self.state]

            # The update applied to the state value
            update = self.learning_rate * \
                     (self.reward_timestep + discounted_value_s_1 - value_s_0)

            # The new state value
            self.state_value[self.last_state] = value_s_0 + update

        # Assign a value to the terminal state
        else:
            value_s_1 = self.state_value[self.state]
            self.state_value[self.state] = value_s_1 + \
                                           self.learning_rate * \
                                           (self.reward_timestep - value_s_1)


    # Not actually a method from Mouse but from FirstVisitStateValueMonteCarloMouse
    # Again indicates the need of a intermediate State Value mouse (or even Mouse)
    def update_policy(self):
        '''
        Update the policy to choose the optimal action
        '''
        for state in self.states:
            best_actions = []
            best_value = 0
            for action in self.actions:
                value = self.state_value[self.transition[state][action]]
                if  value > best_value:
                    best_actions = [action]
                    best_value = value
                elif value == best_value:
                    best_actions.append(action)

            self.policy.loc[state] = [1/len(best_actions) if x in best_actions else 0 for x in self.actions]


    def play_one_round(self):
        '''Extended method of mouse'''
        super().play_one_round()
        self.update_state_values()
        self.update_policy()
