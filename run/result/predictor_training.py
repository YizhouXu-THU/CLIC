import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


path = './log/predictor/predictor_mlp/'
files = os.listdir(path)
train_loss_mlp, valid_loss_mlp = [], []
for file in files:
    if file.startswith('train_loss'):
        train_loss_mlp.append(np.load(path+file))
    elif file.startswith('valid_loss'):
        valid_loss_mlp.append(np.load(path+file))
train_loss_mlp, valid_loss_mlp = np.array(train_loss_mlp), np.array(valid_loss_mlp)
train_loss_mlp_mean, valid_loss_mlp_mean = np.mean(train_loss_mlp, axis=0), np.mean(valid_loss_mlp, axis=0)
train_loss_mlp_std, valid_loss_mlp_std = np.std(train_loss_mlp, axis=0), np.std(valid_loss_mlp, axis=0)

path = './log/predictor/predictor_rnn/'
files = os.listdir(path)
train_loss_rnn, valid_loss_rnn = [], []
for file in files:
    if file.startswith('train_loss'):
        train_loss_rnn.append(np.load(path+file))
    elif file.startswith('valid_loss'):
        valid_loss_rnn.append(np.load(path+file))
train_loss_rnn, valid_loss_rnn = np.array(train_loss_rnn), np.array(valid_loss_rnn)
train_loss_rnn_mean, valid_loss_rnn_mean = np.mean(train_loss_rnn, axis=0), np.mean(valid_loss_rnn, axis=0)
train_loss_rnn_std, valid_loss_rnn_std = np.std(train_loss_rnn, axis=0), np.std(valid_loss_rnn, axis=0)

path = './log/predictor/predictor_lstm/'
files = os.listdir(path)
train_loss_lstm, valid_loss_lstm = [], []
for file in files:
    if file.startswith('train_loss'):
        train_loss_lstm.append(np.load(path+file))
    elif file.startswith('valid_loss'):
        valid_loss_lstm.append(np.load(path+file))
train_loss_lstm, valid_loss_lstm = np.array(train_loss_lstm), np.array(valid_loss_lstm)
train_loss_lstm_mean, valid_loss_lstm_mean = np.mean(train_loss_lstm, axis=0), np.mean(valid_loss_lstm, axis=0)
train_loss_lstm_std, valid_loss_lstm_std = np.std(train_loss_lstm, axis=0), np.std(valid_loss_lstm, axis=0)

plt.figure(figsize=(6, 4))
plt.plot(train_loss_mlp_mean, linewidth=1.2, label='MLP')
plt.fill_between(np.arange(len(train_loss_mlp_mean)), 
                 train_loss_mlp_mean-train_loss_mlp_std, train_loss_mlp_mean+train_loss_mlp_std, 
                 alpha=0.4)
plt.plot(train_loss_rnn_mean, linewidth=1.2, label='RNN')
plt.fill_between(np.arange(len(train_loss_rnn_mean)), 
                 train_loss_rnn_mean-train_loss_rnn_std, train_loss_rnn_mean+train_loss_rnn_std, 
                 alpha=0.4)
plt.plot(train_loss_lstm_mean, linewidth=1.2, label='LSTM')
plt.fill_between(np.arange(len(train_loss_lstm_mean)), 
                 train_loss_lstm_mean-train_loss_lstm_std, train_loss_lstm_mean+train_loss_lstm_std, 
                 alpha=0.4)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./figure/train_loss.pdf')

# plt.figure()
# plt.plot(valid_loss_mlp_mean, linewidth=0.1, label='MLP')
# plt.fill_between(np.arange(len(valid_loss_mlp_mean)), 
#                  valid_loss_mlp_mean-valid_loss_mlp_std, valid_loss_mlp_mean+valid_loss_mlp_std, 
#                  alpha=0.2)
# plt.plot(valid_loss_rnn_mean, linewidth=0.1, label='RNN')
# plt.fill_between(np.arange(len(valid_loss_rnn_mean)), 
#                  valid_loss_rnn_mean-valid_loss_rnn_std, valid_loss_rnn_mean+valid_loss_rnn_std, 
#                  alpha=0.2)
# plt.plot(valid_loss_lstm_mean, linewidth=0.1, label='LSTM')
# plt.fill_between(np.arange(len(valid_loss_lstm_mean)), 
#                  valid_loss_lstm_mean-valid_loss_lstm_std, valid_loss_lstm_mean+valid_loss_lstm_std, 
#                  alpha=0.2)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('./log/predictor/valid_loss.pdf')
