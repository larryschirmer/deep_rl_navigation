import torch
from torch.autograd import Variable
import random
import numpy as np
from matplotlib import pyplot as plt
import math

d_pad = {
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3
}


def train_model(hyperparams, actor_env, training, exp_replay, double_per, metrics):

    (epochs, epsilon, gamma) = hyperparams
    (model, model_, brain_name, env) = actor_env
    (loss_fn, optimizer) = training
    (buffer_size, replay, batch_size) = exp_replay
    (e, a, b, c, c_step) = double_per
    (losses, scores, average_scores) = metrics

    use_GPU = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_GPU else "cpu")
    print(device)

    if (use_GPU):
        device_id = torch.cuda.current_device()
        print(torch.cuda.get_device_name(device_id))

    model = model.to(device)
    model_ = model_.to(device)

    start_epsilon = epsilon
    start_b = b
    epoch_losses = []

    for epoch in range(epochs):
        # reset the environment
        score = 0
        status = 1
        env_info = env.reset(train_mode=True)[brain_name]

        # St
        state_ = env_info.vector_observations
        state = Variable(torch.from_numpy(state_).float()).to(device)

        while(status == 1):
            # copy weights into target net every c iterations
            c_step += 1
            if c_step > c:
                model_.load_state_dict(model.state_dict())
                c_step = 0

            # predicted Q values from current state
            qval = model(state).cpu().data.numpy()

            # select the next action using an epsilon greedy policy
            if (random.random() < epsilon):
                action = np.random.randint(0, 4)
            else:
                action = (np.argmax(qval))

            # send the action to the environment
            env_info = env.step(action)[brain_name]
            next_state_ = env_info.vector_observations      # get the next state
            next_state = Variable(torch.from_numpy(
                next_state_).float()).to(device)  # St(+1)

            reward = env_info.rewards[0]                    # get the reward
            score += reward

            # see if episode has finished
            done = env_info.local_done[0]

            # get the largest expected reward from the target net
            max_Q = np.max(model_(next_state).cpu().data.numpy())
            update = (reward + (gamma * max_Q))

            # get the error and a measure of how surprising it was to the network
            error = np.absolute(qval[0][action] - update)
            priority = (error + e) ** a
            sample_importance = ((1/buffer_size) * (1/priority)) ** b

            # Update replay buffer
            if (len(replay) < buffer_size):
                replay.append(
                    (state, action, reward, next_state, sample_importance))
            else:
                replay.pop(0)
                replay.append(
                    (state, action, reward, next_state, sample_importance))

            # Retrain Model
            if (len(replay) == buffer_size):
                # normalize priority list
                priorities = [states[4] for states in replay]

                # make a randon weighted choice from which experiences to learn from
                mini_batch = random.choices(
                    replay, weights=priorities, k=batch_size)

                X_train = Variable(torch.empty(
                    batch_size, 4, dtype=torch.float))
                y_train = Variable(torch.empty(
                    batch_size, 4, dtype=torch.float))
                h = 0

                # train the network on a batch of saved Action-State-Rewards
                for memory in mini_batch:
                    # new_qval = qval + step * (R(+1) + discount * max_new_Q - qval)

                    old_state_m, action_m, reward_m, new_state_m, priority = memory
                    old_qval = model(old_state_m)
                    new_qval = model(new_state_m).cpu().data.numpy()
                    max_new_Q = np.max(new_qval)

                    y = torch.zeros((1, 4))
                    y[:] = old_qval[:]

                    update_m = (reward_m + (gamma * max_new_Q))

                    y[0][action_m] = update_m
                    X_train[h] = old_qval
                    y_train[h] = y
                    h += 1

                loss = loss_fn(X_train, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            state = next_state

            if done:
                status = 0

        if epsilon > 0.1:
            epsilon = (0.1 - start_epsilon) / \
                (epochs - 0) * epoch + start_epsilon

        if b < 1.:
            b = (1. - start_b) / (epochs - 0) * epoch + start_b

        # print stats
        scores.append(score)
        epoch_loss = 0. if len(epoch_losses) == 0 else np.average(epoch_losses)
        losses.append(epoch_loss)
        average_score = 0. if len(scores) < 101 else np.average(scores[-100:])
        average_scores.append(average_score)
        print("epoch {}, loss: {:.5f}, epsilon: {:.2f}, b: {:.2f}, avg: {:.2f} :: {}".format(
            epoch, 0. if math.isnan(epoch_loss) else epoch_loss, epsilon, b, average_score, score))


def plot_losses(losses, filename):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(np.arange(len(losses)), losses)

    if (filename):
        plt.savefig(filename)


def plot_scores(scores, filename, plotName='Score'):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(plotName)
    plt.xlabel('Episode #')

    if (filename):
        plt.savefig(filename)


def save_model(model, optimizer, replay, filename):
    model = model.cpu()

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'replay': replay
    }
    torch.save(state, filename)


def load_model(model, optimizer, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    replay = checkpoint['replay']

    if evalMode:
        model.eval()
    else:
        model.train()

    return model, optimizer, replay


def test_model(actor_env, attemps, filename):
    (model, brain_name, env) = actor_env

    use_GPU = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_GPU else "cpu")
    print(device)

    model = model.to(device)

    scores = []

    for _ in range(attemps):
        env_info = env.reset(train_mode=True)[
            brain_name]  # reset the environment
        state_ = env_info.vector_observations[0].reshape(1, 37)

        # St
        state = Variable(torch.from_numpy(state_).float()).to(device)
        score = 0                                          # initialize the score

        while True:
            # predicted Q values from current state
            qval = model(state)
            qval_ = qval.cpu().data.numpy()

            # select an action
            action = (np.argmax(qval_))

            # send the action to the environment
            env_info = env.step(action)[brain_name]

            # get the next state
            next_state_ = env_info.vector_observations[0]
            next_state = Variable(torch.from_numpy(
                next_state_).float()).to(device)  # St(+1)

            # get the reward
            reward = env_info.rewards[0]

            # see if episode has finished
            done = env_info.local_done[0]

            # update the score
            score += reward

            # roll over the state to next time step
            state = next_state

            # exit loop if episode finished
            if done:
                break

        scores.append(score)

    scores = np.array(scores)
    average_score = np.average(scores)
    max_score = np.max(scores)

    print("Avg. score: {}, with a high of: {}".format(average_score, max_score))

    plot_scores(scores, filename)
