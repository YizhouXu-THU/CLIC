import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.scenario_lib import scenario_lib
from utils.reward_predictor import reward_predictor


def train_predictor(model: reward_predictor, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=500, lr=1e-3) -> reward_predictor:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        out = model(torch.FloatTensor(X_train))
        y = torch.tensor(y_train)
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def main():
    lib = scenario_lib()
    X_train = lib.data[lib.sample(size=1024)]
    # TODO: use actual y_train
    y_train = np.zeros((X_train.shape[0]))
    predictor = reward_predictor(num_input=lib.max_dim)
    predictor = train_predictor(predictor, X_train, y_train)
    lib.labeling(predictor)
    select_scenario = lib.data[lib.select(size=100)]


if __name__ == '__main__':
    main()
