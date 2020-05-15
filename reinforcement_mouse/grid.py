'''
The module grid defines the environment the mouse operates in.
'''

import pandas as pd
import cv2


class Grid:
    '''
    The class grid defines the states of the game and the possible action
    for each state as well as the payoffs.
    '''

    # Background image of the grid whith the cheese
    image = cv2.imread('../images/mouse_grid.png')
    # cheese = cv2.imread('./images/cheese.svg')
    # double_cheese = cv2.imread('./images/cheese.svg')
    mouse_image = cv2.imread('../images/mouse.png')


    def __init__(self, mouse, size=9):
        self.mouse = mouse

        # Resize the mouse images
        self.mouse_image = self.resize_mouse()

        # The size of the grid
        self.size = size

        # All possible states
        self.states = list(range(1, self.size+1))

        # Mapping between actions and states
        # self.transition = pd.DataFrame([
        #                              [7, 2, 4, 3],
        #                              [8, 3, 5, 1],
        #                              [9, 1, 6, 2],
        #                              [1, 5, 7, 6],
        #                              [2, 6, 8, 4],
        #                              [3, 4, 9, 5],
        #                              [4, 8, 1, 9],
        #                              [5, 9, 2, 7],
        #                              [6, 7, 3, 8]
        #                             ],
        #                             index=self.states,
        #                             columns=['up', 'right', 'down', 'left'])

        self.positions = {
                          1: {'y': 50, 'x': 50},
                          2: {'y': 50, 'x': 270},
                          3: {'y': 50, 'x': 500},
                          4: {'y': 300, 'x': 50},
                          5: {'y': 300, 'x': 270},
                          6: {'y': 300, 'x': 500},
                          7: {'y': 500, 'x': 50},
                          8: {'y': 500, 'x': 270},
                          9: {'y': 500, 'x': 500},
                         }

    def draw(self):
        '''
        The method draw draws the mouse and the cheese into the image.
        '''
        self.frame = self.image.copy()
        y = self.positions[self.mouse.state]['y']
        x = self.positions[self.mouse.state]['x']
        self.frame[y:y+self.mouse_image.shape[0], \
                   x:x+self.mouse_image.shape[1]] = self.mouse_image

    def resize_mouse(self):
        '''
        The method resize_mouse resizes the original mouse image.
        '''

        scale_percent = 20 # percent of original size
        width = int(self.mouse_image.shape[1] * scale_percent / 100)
        height = int(self.mouse_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        return cv2.resize(self.mouse_image, dim, interpolation = cv2.INTER_AREA)
