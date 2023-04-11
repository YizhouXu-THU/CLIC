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
    t0 = time.time()
    lib = scenario_lib()
    predictor = reward_predictor(num_input=lib.max_dim)
    t1 = time.time()
    print('Preparation time: %.1fs' % (t1-t0))

    # 1. Sample
    X_train = lib.data[lib.sample(size=1024)]
    t2 = time.time()
    print('Sampling time: %.1fs' % (t2-t1))

    # 2. Evaluate / 3. Interact
    # TODO: use actual y_train
    y_train = np.zeros((X_train.shape[0]))
    t3 = time.time()
    print('Evaluation time: %.1fs' % (t3-t2))

    # 4. Train reward predictor
    predictor = train_predictor(predictor, X_train, y_train)
    t4 = time.time()
    print('Training reward predictor time: %.1fs' % (t4-t3))

    # 5. Labeling
    lib.labeling(predictor)
    t5 = time.time()
    print('Labeling time: %.1fs' % (t5-t4))

    # 6. Select
    select_scenario = lib.data[lib.select(size=100)]
    t6 = time.time()
    print('Selecting time: %.1fs' % (t6-t5))

    # 7. Train AV model
    # TODO: add AV model
    t7 = time.time()
    print('Training AV model time: %.1fs' % (t7-t6))


if __name__ == '__main__':
    main()
