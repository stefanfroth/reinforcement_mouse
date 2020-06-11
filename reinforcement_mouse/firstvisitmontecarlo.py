'''
Defines a mouse that walks through a grid and searches for cheese. It learns
to find the cheese using reinforecement learning. It uses a first visit monte
carlo method to estimate state values and uses that to choose a better policy.
'''

import pandas as pd
import numpy as np
from collections import defaultdict
from mouse import Mouse

class FirstVisitStateValueMonteCarloMouse(Mouse):
        '''
        Implements the first visit monte carlo method of learning the state
        value function.

        Parameters:
        -----------
        states:            States the mouse can be in, depends on the grid
        starting_position: In which position does the mouse start
        exploration:       True means the mouse is exploring, False means it is not
        discount_factor:   The discount factor of the game
        learning_rate:     Learning rate at which the value is updated
        '''
        def __init__(self, nr_of_sample_draws):
            super().__init__(states=list(range(1, 10)), starting_position=5,
                         exploration=False, discount_factor=0.8,
                         learning_rate=0.5)

            # Nr of samples that are drawn in order to estimate the state value function
            self.nr_of_sample_draws = nr_of_sample_draws

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

            # Save the visited states
            self.state_history = defaultdict(list)

        ### The following methods are only needed for the subclasses

        def choose_action(self):
            '''
            Extends the choose_action method of the parent class
            '''
            # self.last_state = self.state

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

            printable_state_values = {key: round(value, 2)
                                      for key, value
                                      in self.state_value.items()}
            print(f'The state_values are {printable_state_values}')#[self.transition[self.state][action]] for action in self.actions]}')
            # print(f'The best possible actions are {best_action}')


        def transition_to_next_state(self):
            super().transition_to_next_state()

            # Save the new state in the state history
            self.state_history[self.episode].append(self.state)


        def update_state_values(self):
            '''
            Updates the values of the state_value function.
            '''
            # Update (v_{t-1} + alpha * (v_t - v_{t-1}))
            for state in self.states:
                if self.state_occurence[state] > 0:
                    average_reward = self.state_reward[state]/self.state_occurence[state]

                    self.state_value[state] = average_reward


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


        # def reset_state_occurence(self):
        #     '''
        #     Resets the state occurence of the first policy
        #     '''
        #     ...


        def increase_state_reward_occurence(self):
            '''
            Increases the state_reward used to calculate the state_value.
            '''
            for state in set(self.state_history[self.episode]):
                self.state_reward[state] += self.reward_episode
                self.state_occurence[state] += 1

        def transition_to_new_episode(self):
            '''
            Extend parent classes method to update state values
            '''
            print(f'Transitioning to new episode {self.episode+1}')
            self.increase_state_reward_occurence()

            # Update the state_values and the policy after a preset amount of episodes
            if self.episode % self.nr_of_sample_draws == 0:
                self.update_state_values()
                self.update_policy()
                print(f'Episode: {self.episode}; Updating state values')

            # Functionality of the class Mouse
            super().transition_to_new_episode()
            print('Breakpoint')
