from unityagents import UnityEnvironment
import copy

from model import get_model
from helpers import train_model, save_model, plot_losses, plot_scores, test_model, load_model

# hyperparameters
lr = 0.0001
gamma = 0.9

batch_size = 30
buffer_size = 5000

c = 500
c_step = 0
e = 0.01
a = 0.6

input_depth = 37
hidden0 = 500
hidden1 = 250
hidden2 = 100
output_depth = 4

replay = []

model, loss_fn, optimizer = get_model(
    input_depth, hidden0, hidden1, hidden2, output_depth, lr)

filename = 'checkpoint-1800-2.pt'
model, optimizer, replay = load_model(model, optimizer, filename)

model_ = copy.deepcopy(model)

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# train model

epochs = 700
epsilon = 0.3  # decays over the course of training
losses = []
scores = []

hyperparams = (epochs, epsilon, gamma)
actor_env = (model, model_, brain_name, env)
training = (loss_fn, optimizer)
exp_replay = (buffer_size, replay, batch_size)
double_per = (e, a, c, c_step)
metrics = (losses, scores)

train_model(hyperparams, actor_env, training, exp_replay,
            double_per, metrics, manual_override=False)

plot_losses(losses, 'losses-2500-2.png')
plot_scores(scores, 'scores-2500-2.png')

save_model(model, optimizer, replay, 'checkpoint-2500-2.pt')

test_actor_env = (model, brain_name, env)
attemps = 100
filename = 'test_scores-2500-2.png'

test_model(test_actor_env, attemps, filename)
