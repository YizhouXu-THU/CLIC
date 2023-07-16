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
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import evaluate, train_valid_predictor


# Prepare
eval_size = 4096
batch_size = 128
train_size = 128
epochs = 200
learning_rate = 1e-4
sumo_gui = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('eval_size:', eval_size, ' learning_rate:', learning_rate)

# lib = scenario_lib(path='./scenario_lib/', 
#                    npy_path='./all_data.npy')
# pred = predictor(num_input=lib.max_dim, device=device)
# pred.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(pred.parameters(), lr=learning_rate)
# env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
# av_model = RL_brain(env, capacity=0, device=device)

# all_label = evaluate(av_model, env, scenarios=lib.data)
# success_rate = 1 - np.sum(all_label) / all_label.size
# print('Success rate: %.3f' % success_rate)

# all_data = np.append(lib.data, all_label.reshape(-1,1), axis=1)
# np.save('./all_data.npy', all_data)
# env.close()

lib = scenario_lib(path='./scenario_lib/', 
                   npy_path='./all_data.npy')
all_data = np.load('./all_data.npy')
X = all_data[:, :-1]
y = all_data[:, -1].reshape(-1)
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

pred = predictor(num_input=max_dim, device=device)
pred.to(device)

pred = train_valid_predictor(pred, X_train, y_train, X_test, y_test, 
                             epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)

lib.labeling(pred)
lib.visualize_label_distribution(train_size, save_path=None)
