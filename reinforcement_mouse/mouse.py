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

        ### Cutoff: The rest of the __init__ function belongs to subclasses

        # Create the action-value function
        self.q = pd.DataFrame(np.zeros(shape=(9,4)),
                              columns=['up', 'right', 'left', 'down'],
                              index=self.states)




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
        self.reward_timestep = self.reward_function[self.state] * \
                               self.discount_factor**self.timestep
        self.reward_episode += self.reward_timestep
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


## First implement that it plays a couple of rounds in order to estimate the
# state_values
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

            print(f'The actions are {self.actions}')
            print(f'The state_values are {self.state_value}')#[self.transition[self.state][action]] for action in self.actions]}')
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


        def reset_state_occurence(self):
            '''
            Resets the state occurence of the first policy
            '''
            ...


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


    def update_state_values(self):
        '''
        Update the state values
        '''
        self.state_value
        


class PolicyIterationMouse(Mouse):
    ...

class ValueIterationMouse(Mouse):
    ...

class OnPolicyFirstVisitMCControl(Mouse):
    '''
    P. 101/123 book
    '''
    ...
