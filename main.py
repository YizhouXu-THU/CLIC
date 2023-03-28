import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import scenario_lib
from utils import reward_predictor


def train_predictor(model: reward_predictor, X_train: np.ndarray, y_train: np.ndarray, epochs=500) -> reward_predictor:
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        out = model(torch.FloatTensor(X_train))
        y = torch.tensor(y_train, dtype=torch.int64)
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def main():
    lib = scenario_lib()
    X_train = lib.data[lib.sample(batch_size=3)]


if __name__ == '__main__':
    main()
