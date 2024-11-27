# Snake Reinforcement Learning

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take. In this code we will be focusing on DeepQLearningAgent only.

## Code Structure
[game_environment.py](../game_environment.py) contains the necessary code to create and interact with the snake environment (class Snake and SnakeNumpy). The interface is similar to openai gym interface.
Key points for SnakeNumpy Class
* Use the games argument to decide the number of games to play in parallel
* Set frame_mode to True for continuously running the game, any completed game is immediately reset
* When performing reset, use the stateful argument to decide whether to do a hard reset or not

[agent.py](../agent.py) contains the agent for playing the game. It implements and trains a convolutional neural network for the action values. Following classes are available
<table>
    <head>
        <tr>
        <th> Class </th><th> Description</th>
        </tr>
    </head>
    <tr><td>DeepQLearningAgent/td><td>Deep Q Learning Algorithm with CNN Network</td></tr>
    <tr><td>PolicyGradientAgent</td><td>Policy Gradient Algorithm with CNN Network</td></tr>
    <tr><td>AdvantageActorCriticAgent</td><td>Advantage Actor Critic (A2C) Algorithm with CNN Network</td></tr>
    <tr><td>HamiltonianCycleAgent</td><td>Creates a Hamiltonian Cycle on Even Sized Boards for Traversal</td></tr>
    <tr><td>SupervisedLearningAgent</td><td>Trains Using Examples from another Agent/Human</td></tr>
    <tr><td>BreadthFirstSearchAgent</td><td>Repeatedly Finds Shortest Path from Snake Head to Food for Traversal</td></tr>
</table>

[training.py](../training.py) contains the complete code to train an agent.

[game_visualization.py](../game_visualization.py) contains the code to convert the game to mp4 format.

```python
from game_environment import SnakeNumpy
from agent import QLearningAgent
import numpy as np

game_count = 10

env = Snake(board_size=10, frames=2, 
            max_time_limit=298, games=game_count, # Allows running 10 games in parallel
            frame_mode=False) # Allows continuous run of successive games
state = env.reset(stateful=True) # first manual reset required to initialize few variables
agent = QLearningAgent(board_size=10, frames=2, n_actions=env.get_num_actions(),
                       buffer_size=10000)
done = np.zeros((game_count,), dtype=np.uint8)
total_reward = np.zeros((game_count,), dtype=np.uint8)
epsilon = 0.1
while(not done.all()):
    legal_moves = env.get_legal_moves()
    if(np.random.random() <= epsilon):
        action = np.random.choice(np.arange(env.get_num_actions(), game_count)
    else:
        action = agent.move(s, legal_moves, values=env.get_values())
    next_state, reward, done, info, next_legal_moves = env.step(action)
    # info contains time, food (food count), termination_reason (if ends)
    agent.add_to_buffer([state, action, reward, next_state, done, next_legal_moves])
    total_reward += reward
    state = next_state.copy()
agent.train_agent(batch_size=32) # perform one step of gradient descent
agent.update_target_net() # update the target network


# another way to use the environment is the frame mode
# which allows faster accumulation of training data
env = Snake(board_size=10, frames=2, 
            max_time_limit=298, games=game_count,
            frame_mode=True)
while(True):
    s = env.reset(stateful=True)
    total_frames = 0
    while(total_frames < 100):
        """ same code as above """
        total_frames += game_count
    """ add data to buffer """
```

## Experiments
Configuration for different experiments can be found in [model_versions.json](../model_versions.json) file.

## Environment with Obstacles
40000 10x10 boards are randomly generated with 8 cells as obstacles, while ensuring that the snake always has a path to navigate through the board. The code for the same can be located at : [obstacles_board_generator.py](../obstacles_board_generator.py)<br>
Based on the sample plays below, it is evident that the learned policy generalizes well over boards with random obstacles, and even works good on boards with higher number of obstacles (although it has a higher chance of getting stuck in a loop)<br>

Sample games from the best [model](models\v17.1\model_163500.h5) with obstacles using PyTorch<br><img width="400" height="400" src="https://github.com/yan008uit/GA02/blob/main/images/game_visual_v17.1_163500_14_ob_4.gif" alt="model v17.1 agent"><img width="400" height="400" src="https://github.com/yan008uit/GA02/blob/main/images/game_visual_v17.1_163500_14_ob_3.gif" alt="model v17.1 agent"><br>

More sample games from the [model](models\v17.1\model_163500.h5) with obstacles using PyTorch<br><img width="400" height="400" src="https://github.com/yan008uit/GA02/blob/main/images/game_visual_v17.1_163500_14_ob_2.gif" alt="model v17.1 agent"><img width="400" height="400" src="https://github.com/yan008uit/GA02/blob/main/images/game_visual_v17.1_163500_14_ob_1.gif" alt="model v17.1 agent"><br>

<img width="400" height="400" src="https://github.com/yan008uit/GA02/blob/main/images/game_visual_v17.1_163500_14_ob_0.gif" alt="model v17.1 agent">