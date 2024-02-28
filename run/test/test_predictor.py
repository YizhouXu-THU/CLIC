import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ptflops import get_model_complexity_info

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_mlp, predictor_rnn, predictor_lstm, predictor_vae
from utils.function import set_random_seed, train_validate_predictor, train_validate_predictor_vae, train_predictor_vae


eval_size = 4096
batch_size = 128
train_size = 128
epochs = 20
epochs_vae = 1000
learning_rate = 1e-4
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42    # 14, 42, 51, 71, 92
set_random_seed(random_seed)
# print('eval_size:', eval_size, ' learning_rate:', learning_rate)

lib = scenario_lib(path='./data/all.npz')
npz_data = np.load('./data/all.npz', allow_pickle=True)
X = npz_data['scenario']
y = npz_data['label']
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# predictor = predictor_mlp(input_dim=max_dim, device=device)
# predictor = predictor_rnn(timestep=lib.max_timestep, input_dim=max_dim, device=device)
# predictor = predictor_lstm(timestep=lib.max_timestep, input_dim=max_dim, device=device)
# predictor.to(device)
# flops, params = get_model_complexity_info(predictor, (max_dim,), print_per_layer_stat=False)
# print('Flops:', flops, ' Params:', params)
# train_validate_predictor(predictor, X_train, y_train, X_test, y_test, 
#                          epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)
# lib.labeling(predictor)

vae = predictor_vae(input_dim=max_dim, device=device)
classifier = predictor_mlp(input_dim=vae.latent_dim, device=device)
vae.to(device), classifier.to(device)
flops_vae, params_vae = get_model_complexity_info(vae, (max_dim,), print_per_layer_stat=False)
flops_classifier, params_classifier = get_model_complexity_info(classifier, (vae.latent_dim,), print_per_layer_stat=False)
print('VAE Flops:', flops_vae, 'VAE Params:', params_vae, 
      'Classifier Flops:', flops_classifier, 'Classifier Params:', params_classifier)
train_validate_predictor_vae(vae, classifier, lib, X_train, y_train, X_test, y_test, 
                             epochs_vae=epochs_vae, epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)
lib.labeling_vae(vae, classifier)

# lib.visualize_label_distribution(num_select=train_size, num_sample=eval_size)
