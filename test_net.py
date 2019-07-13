from unityagents import UnityEnvironment
from time import perf_counter
import numpy as np
import copy

from model import get_model
from helpers import train_model, save_model, plot_losses, plot_scores, test_model, load_model

# hyperparameters
lr = 0.0005
gamma = 0.9

batch_size = 10
buffer_size = 5000

c = 750
c_step = 0
e = 0.01
a = 0.6
b = 0.4
input_depth = 37
hidden0 = 128
hidden1 = 128
hidden2 = 128
output_depth = 4

replay = []

model, loss_fn, optimizer = get_model(
    input_depth, hidden0, hidden1, hidden2, output_depth, lr)

filename = 'checkpoint-2000.pt'
model, optimizer, replay = load_model(model, optimizer, filename)

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

test_actor_env = (model, brain_name, env)
attemps = 100
filename = 'test_scores-{}.png'.format(attemps)

test_model(test_actor_env, attemps, filename, viewableSpeed=False)
