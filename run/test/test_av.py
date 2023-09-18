import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime
import wandb
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.av_policy import RL_brain
from utils.environment import Env
from utils.function import evaluate, train_av_online, train_av


eval_size = 4096
batch_size = 128
train_size = 128
epochs = 20
episodes = 10
learning_rate = 1e-4
auto_alpha = True
use_wandb = False
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate)

if use_wandb:
    wandb_config = {
        'batch_size': batch_size, 
        'train_size': train_size, 
        'episodes': episodes, 
        'learning_rate': learning_rate, 
        'auto_alpha': auto_alpha, 
    }
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=datetime.now().strftime('%Y%m%d-%H%M')+'-test_av', # for example: '20230509-1544-test_av'
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

index = lib.sample(size=train_size)
train_scenario = lib.data[index]

all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate before training: %.3f' % success_rate)

init_policynet_params = av_model.policy_net.state_dict()

av_model = train_av_online(av_model, env, train_scenario, episodes, auto_alpha, wandb_logger)
# av_model = train_av(av_model, env, train_scenario, epochs, episodes, auto_alpha, wandb_logger)

all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate after training: %.3f' % success_rate)

av_model.policy_net.load_state_dict(init_policynet_params)
all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate using loaded params: %.3f' % success_rate)

env.close()
