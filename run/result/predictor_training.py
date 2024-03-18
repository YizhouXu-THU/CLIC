import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

alpha_train = 10
alpha_valid = 50

path = './log/predictor/predictor_mlp/'
files = os.listdir(path)
train_loss_mlp, valid_loss_mlp = [], []
for file in files:
    if file.startswith('train_loss'):
        # train_loss_mlp.append(np.load(path+file))
        train_loss = np.load(path+file)
        train_loss_mlp.append(np.convolve(train_loss, np.ones(alpha_train)/alpha_train, mode='valid'))
    elif file.startswith('valid_loss'):
        # valid_loss_mlp.append(np.load(path+file))
        valid_loss = np.load(path+file)
        valid_loss_mlp.append(np.convolve(valid_loss, np.ones(alpha_valid)/alpha_valid, mode='valid'))
train_loss_mlp, valid_loss_mlp = np.array(train_loss_mlp), np.array(valid_loss_mlp)
train_loss_mlp_mean, valid_loss_mlp_mean = np.mean(train_loss_mlp, axis=0), np.mean(valid_loss_mlp, axis=0)
train_loss_mlp_std, valid_loss_mlp_std = np.std(train_loss_mlp, axis=0), np.std(valid_loss_mlp, axis=0)
# train_loss_mlp_mean = np.convolve(train_loss_mlp_mean, np.ones(alpha)/alpha, mode='valid')
# train_loss_mlp_std = np.convolve(train_loss_mlp_std, np.ones(alpha)/alpha, mode='valid')
# valid_loss_mlp_mean = np.convolve(valid_loss_mlp_mean, np.ones(alpha)/alpha, mode='valid')
# valid_loss_mlp_std = np.convolve(valid_loss_mlp_std, np.ones(alpha)/alpha, mode='valid')

path = './log/predictor/predictor_rnn/'
files = os.listdir(path)
train_loss_rnn, valid_loss_rnn = [], []
for file in files:
    if file.startswith('train_loss'):
        # train_loss_rnn.append(np.load(path+file))
        train_loss = np.load(path+file)
        train_loss_rnn.append(np.convolve(train_loss, np.ones(alpha_train)/alpha_train, mode='valid'))
    elif file.startswith('valid_loss'):
        # valid_loss_rnn.append(np.load(path+file))
        valid_loss = np.load(path+file)
        valid_loss_rnn.append(np.convolve(valid_loss, np.ones(alpha_valid)/alpha_valid, mode='valid'))
train_loss_rnn, valid_loss_rnn = np.array(train_loss_rnn), np.array(valid_loss_rnn)
train_loss_rnn_mean, valid_loss_rnn_mean = np.mean(train_loss_rnn, axis=0), np.mean(valid_loss_rnn, axis=0)
train_loss_rnn_std, valid_loss_rnn_std = np.std(train_loss_rnn, axis=0), np.std(valid_loss_rnn, axis=0)
# train_loss_rnn_mean = np.convolve(train_loss_rnn_mean, np.ones(alpha)/alpha, mode='valid')
# train_loss_rnn_std = np.convolve(train_loss_rnn_std, np.ones(alpha)/alpha, mode='valid')
# valid_loss_rnn_mean = np.convolve(valid_loss_rnn_mean, np.ones(alpha)/alpha, mode='valid')
# valid_loss_rnn_std = np.convolve(valid_loss_rnn_std, np.ones(alpha)/alpha, mode='valid')

path = './log/predictor/predictor_lstm/'
files = os.listdir(path)
train_loss_lstm, valid_loss_lstm = [], []
for file in files:
    if file.startswith('train_loss'):
        # train_loss_lstm.append(np.load(path+file))
        train_loss = np.load(path+file)
        train_loss_lstm.append(np.convolve(train_loss, np.ones(alpha_train)/alpha_train, mode='valid'))
    elif file.startswith('valid_loss'):
        # valid_loss_lstm.append(np.load(path+file))
        valid_loss = np.load(path+file)
        valid_loss_lstm.append(np.convolve(valid_loss, np.ones(alpha_valid)/alpha_valid, mode='valid'))
train_loss_lstm, valid_loss_lstm = np.array(train_loss_lstm), np.array(valid_loss_lstm)
train_loss_lstm_mean, valid_loss_lstm_mean = np.mean(train_loss_lstm, axis=0), np.mean(valid_loss_lstm, axis=0)
train_loss_lstm_std, valid_loss_lstm_std = np.std(train_loss_lstm, axis=0), np.std(valid_loss_lstm, axis=0)
# train_loss_lstm_mean = np.convolve(train_loss_lstm_mean, np.ones(alpha)/alpha, mode='valid')
# train_loss_lstm_std = np.convolve(train_loss_lstm_std, np.ones(alpha)/alpha, mode='valid')
# valid_loss_lstm_mean = np.convolve(valid_loss_lstm_mean, np.ones(alpha)/alpha, mode='valid')
# valid_loss_lstm_std = np.convolve(valid_loss_lstm_std, np.ones(alpha)/alpha, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(train_loss_mlp_mean, linewidth=1.2, label='MLP')
axes[0].fill_between(np.arange(len(train_loss_mlp_mean)), 
                     train_loss_mlp_mean-train_loss_mlp_std, train_loss_mlp_mean+train_loss_mlp_std, 
                     alpha=0.2)
axes[0].plot(train_loss_rnn_mean, linewidth=1.2, label='RNN')
axes[0].fill_between(np.arange(len(train_loss_rnn_mean)), 
                     train_loss_rnn_mean-train_loss_rnn_std, train_loss_rnn_mean+train_loss_rnn_std, 
                     alpha=0.2)
axes[0].plot(train_loss_lstm_mean, linewidth=1.2, label='LSTM')
axes[0].fill_between(np.arange(len(train_loss_lstm_mean)), 
                     train_loss_lstm_mean-train_loss_lstm_std, train_loss_lstm_mean+train_loss_lstm_std, 
                     alpha=0.2)
axes[0].set_ylim([0.2, 0.8])
axes[0].tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid()

axes[1].plot(valid_loss_mlp_mean, linewidth=1, label='MLP')
axes[1].fill_between(np.arange(len(valid_loss_mlp_mean)), 
                     valid_loss_mlp_mean-valid_loss_mlp_std, valid_loss_mlp_mean+valid_loss_mlp_std, 
                     alpha=0.2)
axes[1].plot(valid_loss_rnn_mean, linewidth=1, label='RNN')
axes[1].fill_between(np.arange(len(valid_loss_rnn_mean)), 
                     valid_loss_rnn_mean-valid_loss_rnn_std, valid_loss_rnn_mean+valid_loss_rnn_std, 
                     alpha=0.2)
axes[1].plot(valid_loss_lstm_mean, linewidth=1, label='LSTM')
axes[1].fill_between(np.arange(len(valid_loss_lstm_mean)), 
                     valid_loss_lstm_mean-valid_loss_lstm_std, valid_loss_lstm_mean+valid_loss_lstm_std, 
                     alpha=0.2)
axes[1].set_ylim([0.2, 0.8])
axes[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
axes[1].set_xlabel('Iteration')
# axes[1].set_ylabel('Loss')
axes[1].set_title('Validation Loss')
# axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('./figure/predictor_loss.pdf')
