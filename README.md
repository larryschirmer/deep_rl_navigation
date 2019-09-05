# Environment Navigation with Deep Reinforcement Learning

![](https://github.com/larryschirmer/deep_rl_navigation/raw/master/solved_navigator.gif)

## Important files

- This `README.md`: describes the project and its files in detail
- `Navigation.ipynb`: working demonstration of the final trained modal
- `Report.ipynb`: notebook containing the training methods
- `checkpoint-1500.pt`: trained modal checkpoint
- `main.py`: python file used to develop and train network
- `helpers.py`: collection of functions used to train, test, and monitor model 
- `model.py`: functions to return new model, optimizer, and loss function
    - Files from the first successful training run
        - `checkpoint-2000.pt`
        - `ave-scores-2000.png`
        - `scores-2000.png`
        - `losses-2000.png`

## The Environment

The environment that this model solves is a Unity game with a discrete action space. Using:

- 0 - move forward
- 1 - move backward
- 2 - turn left
- 3 - turn right

The agent navigates the environment picking up yellow bananas and avoiding blue bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the environment is for the agent to collect as many yellow bananas as possible while avoiding all the blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with a ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## How to Install

Because one of the project dependencies requires a specific version of tensorflow only available in python 3.5 and earlier, its easiest to use conda to build the environment for this project.

Run the following command to build this environment using the same dependancies I used:

```bash
conda env create -f environment.yml python=3.5
```

[See the conda docs for installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

As for the game engine, select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the project folder of this repository, and unzip (or decompress) the file.

## Algorithms and Methods

The base code for this projects comes from the grid world implementation of "Deep Q-Learning" in chapter 3 of [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action?query=deep%20reinforcement), however many of the advanced techniques such at importance sampling and using a separate target network come from the Udacity's [Deep Reinforcement Learning Class](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

In deep q-learning, a deep neural network takes the state of the environment and returns the expected value for all the actions in the environment's action space. The goal being for the agent to navigate to where the rewards are in the most efficient manor. 

The models handles this prediction with a simple feedforward network written in pytorch. Using just a couple of linear layers with relu activation, the model is able to learn the relationship between the current state and the most valuable action to take. 

```python
# model.py Line: 7
model = Sequential(
    Linear(input_depth, hidden0),
    ReLU(),
    Linear(hidden0, hidden1),
    ReLU(),
    Linear(hidden1, hidden2),
    ReLU(),
    Linear(hidden2, output_depth)
)

loss_fn = MSELoss(reduction='sum')
optimizer = Adam(model.parameters(), lr=lr)
```

While it learns, and even after it is fully trained, an action is chosen at random from the action space to promote exploration. This value (epsilon), manages how greedy the network is allowed to be. 

```python
# helpers.py Line: 49
if (random.random() < epsilon):
    action = np.random.randint(0, 4)
else:
    action = (np.argmax(qval))
```

After each episode, the importance of each result is rated by how wrong/surprised the network was by the result. The value is stored with the other data from the episode and used to weigh which trajectories to batch for training.

```python
# helpers.py Line: 70
error = np.absolute(qval[0][action] - update)
priority = (error + e) ** a
sample_importance = ((1/buffer_size) * (1/priority)) ** b
```

To improve the models ability to predict the max future reward for the next state, a second model is used. This second model is updated after a set number of time steps to keep the predictions stable enough to learn from. The stability of the second model's parameters create a fixed target for the first model to train towards.

```python
# helpers.py Line: 41
if c_step > c:
    model_.load_state_dict(model.state_dict())
    c_step = 0
```

Batching the past experiences after each time step does a couple of things for the agent (model). Learning from the sum of the experiences of each batch helps the agent find patterns to learn from. These experiences can also mitigate regressive behavior patterns from rare one off successes. Batching experiences also benefits the agent by offering more opportunity to review surprising predictions to learn and make better predictions when that state appears again. The overall result ends up being a smoother learning curve and higher rewards every episode.

## Future Work

In the future I could improve this project in two ways. 

### Use convolutional layers to process game pixels

First, the agent is trained to take in and act on a preprocessed version of its environment. This really limits the application of the model to just environments with this specific state space. By adapting the first layers of the model to accept game pixels as input, a model could be reused to learn other games.

### Implement dueling DQN in the training process

Secondly, the agent right now predicts which action will result in the highest future reward for the remainder of the episode, but it makes no distinction as to how valuable each time step is. It has been shown that providing the model with power to tell how valuable each time step is (whether it is about to get a reward or when any action will result in no reward), helps the agent achieve higher rewards sooner.