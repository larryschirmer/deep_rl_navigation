import torch
from torch.nn import Sequential, Linear, ReLU, MSELoss, Dropout
from torch.optim import Adam


def get_model(input_depth, hidden0, hidden1, hidden2, output_depth, lr):
    use_GPU = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_GPU else "cpu")
    
    model = Sequential(
        Linear(input_depth, hidden0),
        ReLU(),
        Linear(hidden0, hidden1),
        ReLU(),
        Linear(hidden1, hidden2),
        ReLU(),
        Linear(hidden2, output_depth)
    )
    
    model = model.to(device)

    loss_fn = MSELoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=lr)

    return model, loss_fn, optimizer
