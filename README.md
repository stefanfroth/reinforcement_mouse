# reinforcement_mouse
The repository reinforcement_mouse contains a simple Python implementation of a mouse that is trained via reinforcement learning to find cheese.

The mouse can choose different algorithms to learn the location of the cheese 🧀 

## Usage

First clone the repository, then pip install the requirements from the requirements.txt. 
Afterwards go to the directory ``reinforcement_mouse/reinforcement_mouse`` and run the following ``python run.py``.


## Algorithms

### Monte Carlo Methods

#### First-visit Monte Carlo Method

In the first-visit Monte Carlo Method the mouse randomly walks over the grid until it finds one of the two locations containing cheese 🧀 . After playing some periods the mouse averages the amount of returns received after a certain state was visited (once) and takes that as an estimate of the value of a certain grid cell (state). Thereafter it chooses the grid cell with the highest value and follows the path of optimal grid cells.

To use the First-Visit Monte Carlo Method to find the cheese 🧀  run ``python run.py -m FVSMC``. Additionally, you can adjust the number of initial random draws by using the ``-d`` flag and choosing the number of draws. The default is ``5``.

### Temporal-Difference Learning

#### Q-Learning

Implementation following Sutton and Barto.

To use the Q-Learning Method to find the cheese 🧀  run ``python run.py -m QLearning``. Additionally you can turn on exploration by providing the ``-e`` flag.

#### Sarsa

Implementation following Sutton and Barto.

To use the Sarsa Method to find the cheese 🧀  run ``python run.py -m Sarsa``. Additionally you can turn on exploration by providing the ``-e`` flag.
