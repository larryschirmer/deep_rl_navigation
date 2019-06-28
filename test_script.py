# dropped epsilon limit, turned up learning rate
from unityagents import UnityEnvironment
import copy

from model import get_model
from helpers import train_model, save_model, plot_losses, plot_scores, test_model, load_model

# hyperparameters
lr = 0.001
gamma = 0.9

batch_size = 10
buffer_size = 5000

c = 750
c_step = 0
e = 0.01
a = 0.5

input_depth = 37
hidden0 = 128
hidden1 = 128
hidden2 = 128
output_depth = 4

replay = []

model, loss_fn, optimizer = get_model(input_depth, hidden0, hidden1, hidden2, output_depth, lr)

# filename = '0-1000-checkpoint.pt'
# model, optimizer, replay = load_model(model, optimizer, filename)

model_ = copy.deepcopy(model)

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# train model

epochs = 3000
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

save_model(model, optimizer, replay, '0-3000-checkpoint.pt')

plot_losses(losses, 'losses-0-3000.png')
plot_scores(scores, 'scores-0-3000.png')
plot_scores(average_scores, 'scores-0-3000.png')

test_actor_env = (model, brain_name, env)
attemps = 100
filename = 'test_scores-0-3000.png'

test_model(test_actor_env, attemps, filename)
