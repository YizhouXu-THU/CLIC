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
from utils.function import evaluate


def main():
    # Prepare
    eval_size = 4096
    epochs = 100
    learning_rate = 1e-4
    sumo_gui = False
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    lib = scenario_lib(path='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/scenario_lib/')
    pred = predictor(num_input=lib.max_dim, device=device)
    pred.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pred.parameters(), lr=learning_rate)
    env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/config/lane.sumocfg', gui=sumo_gui)
    av_model = SAC(env, device=device)
    
    all_label = evaluate(av_model, env, scenarios=lib.data)
    success_rate = 1 - np.sum(all_label) / all_label.size
    print('    Success rate: %.3f' % success_rate)

    test_size = lib.total_num - eval_size
    X_train, X_test, y_train, y_test = train_test_split(lib.data, all_label, test_size=test_size)
    
    for epoch in range(epochs):
        pred.train()
        out = pred(torch.tensor(X_train, dtype=torch.float32, device=device))
        y_train = torch.tensor(y_train, dtype=torch.int64, device=device)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred = torch.max(F.softmax(out, dim=1), dim=1)[1].data.cpu().numpy().squeeze()
        y_train = y_train.data.cpu().numpy().squeeze()
        accuracy = sum(y_pred == y_train) / eval_size
        print('Epoch:', epoch+1, ' train loss: %.4f' % loss.item(), ' train accuracy: %.4f' % accuracy, end='    ')
        
        
        with torch.no_grad():
            pred.eval()
            out = pred(torch.tensor(X_test, dtype=torch.float32, device=device))
            y_test = torch.tensor(y_test, dtype=torch.int64, device=device)
            loss = criterion(out, y_test)
        
        y_pred = torch.max(F.softmax(out, dim=1), dim=1)[1].data.cpu().numpy().squeeze()
        y_test = y_test.data.cpu().numpy().squeeze()
        accuracy = sum(y_pred == y_test) / test_size
        print('test loss: %.4f' % loss.item(), ' test accuracy: %.4f' % accuracy)
    
    env.close()


if __name__ == '__main__':
    main()
