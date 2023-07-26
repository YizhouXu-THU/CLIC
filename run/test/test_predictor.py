import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor
from utils.predictor_rnn import predictor_rnn
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import evaluate, train_valid_predictor


# Prepare
eval_size = 4096
batch_size = 128
train_size = 128
epochs = 20
learning_rate = 1e-4
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# print('eval_size:', eval_size, ' learning_rate:', learning_rate)

lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
all_data = np.load('./data/all.npy')
X = all_data[:, :-1]
y = all_data[:, -1].reshape(-1)
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

pred = predictor(num_input=max_dim, device=device)
# pred = predictor_rnn(timestep=lib.max_timestep, num_input=max_dim, device=device)
pred.to(device)

pred = train_valid_predictor(pred, X_train, y_train, X_test, y_test, 
                             epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)

lib.labeling(pred)
lib.visualize_label_distribution(num_select=train_size)
