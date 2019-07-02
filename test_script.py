# dropped epsilon limit, turned up learning rate
from unityagents import UnityEnvironment
import numpy as np
import copy

from model import get_model
from helpers import train_model, save_model, plot_losses, plot_scores, test_model, load_model

# hyperparameters
lr = 0.0005
gamma = 0.98

batch_size = 20
buffer_size = 5000

c = 1000
c_step = 0
e = 0.01
a = 0.7

input_depth = 37
hidden0 = 256
hidden1 = 256
hidden2 = 256
output_depth = 4

replay = np.array([[0,0,0,0,0]])

model, loss_fn, optimizer = get_model(input_depth, hidden0, hidden1, hidden2, output_depth, lr)

# filename = '0-1000-checkpoint.pt'
# model, optimizer, replay = load_model(model, optimizer, filename, evalMode=False)

model_ = copy.deepcopy(model)

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# train model

epochs = 1000
epsilon = 1.0  # decays over the course of training
losses = []
scores = []
average_scores = []

hyperparams = (epochs, epsilon, gamma)
actor_env = (model, model_, brain_name, env)
training = (loss_fn, optimizer)
exp_replay = (buffer_size, replay, batch_size)
double_per = (e, a, c, c_step)
metrics = (losses, scores, average_scores)

train_model(hyperparams, actor_env, training, exp_replay,
            double_per, metrics, manual_override=False)

save_model(model, optimizer, replay, 'checkpoint-{}.pt'.format(epochs))

plot_losses(losses, 'losses-{}.png'.format(epochs))
plot_scores(scores, 'scores-{}.png'.format(epochs))
plot_scores(average_scores, 'scores-{}.png'.format(epochs), 'Ave Score')

test_actor_env = (model, brain_name, env)
attemps = 100
filename = 'test_scores-{}.png'.format(epochs)

test_model(test_actor_env, attemps, filename)
