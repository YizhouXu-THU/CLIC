import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn, predictor_rnn, predictor_vae
from utils.function import train_validate_predictor, train_validate_predictor_vae


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

predictor = predictor_dnn(input_dim=max_dim, device=device)
# predictor = predictor_rnn(timestep=lib.max_timestep, input_dim=max_dim, device=device)
# predictor = predictor_vae(input_dim=max_dim, device=device)
predictor.to(device)

train_validate_predictor(predictor, X_train, y_train, X_test, y_test, 
                         epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)
# train_validate_predictor_vae(predictor, X_train, y_train, X_test, y_test, 
#                              epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)

lib.labeling(predictor)
lib.visualize_label_distribution(num_select=train_size, num_sample=eval_size)