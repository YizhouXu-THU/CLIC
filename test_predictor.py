import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor
from utils.environment import Env
from utils.av_policy import SAC
from utils.function import evaluate, train_predictor


# Prepare
eval_size = 4096
batch_size = 128
epochs = 100
learning_rate = 1e-4
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# lib = scenario_lib(path='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/scenario_lib/', 
#                    npy_path='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/all_data.npy')
# pred = predictor(num_input=lib.max_dim, device=device)
# pred.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(pred.parameters(), lr=learning_rate)
# env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/config/lane.sumocfg', gui=sumo_gui)
# av_model = SAC(env, device=device)

# all_label = evaluate(av_model, env, scenarios=lib.data)
# success_rate = 1 - np.sum(all_label) / all_label.size
# print('Success rate: %.3f' % success_rate)

# all_data = np.append(lib.data, all_label.reshape(-1,1), axis=1)
# np.save('/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/all_data.npy', all_data)

all_data = np.load('/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/all_data.npy')
X = all_data[:, :-1]
y = all_data[:, -1].reshape(-1)
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

pred = predictor(num_input=max_dim, device=device)
pred.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(pred.parameters(), lr=learning_rate)
optimizer = optim.SGD(pred.parameters(), lr=learning_rate, momentum=0.9)

# pred = train_predictor(pred, X_train, y_train, epochs=epochs, lr=learning_rate, 
#                        batch_size=batch_size, device=device)

total_size = y_train.size
batch_num = math.ceil(total_size/batch_size)
for epoch in range(epochs):
    total_loss = 0.0
    total_correct = 0
    # shuffle
    data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
    np.random.shuffle(data_train)
    
    for iteration in range(batch_num):
        X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
        y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.int64, device=device)
        
        out = pred(X)
        loss = criterion(out, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred = torch.max(out, dim=1)[1].data.cpu().numpy().squeeze()
        y = y.data.cpu().numpy().squeeze()
        total_correct += sum(y_pred == y)
    
    accuracy = total_correct / total_size
    print('Epoch:', epoch+1, ' train loss: %.4f' % (total_loss/batch_num), ' train accuracy: %.4f' % accuracy, end='    ')
    
    with torch.no_grad():
        pred.eval()
        out = pred(torch.tensor(X_test, dtype=torch.float32, device=device))
        y_test = torch.tensor(y_test, dtype=torch.int64, device=device)
        loss = criterion(out, y_test)
    
    y_pred = torch.max(out, dim=1)[1].data.cpu().numpy().squeeze()
    y_test = y_test.data.cpu().numpy().squeeze()
    accuracy = sum(y_pred == y_test) / test_size
    print('test loss: %.4f' % loss.item(), ' test accuracy: %.4f' % accuracy)

# env.close()
