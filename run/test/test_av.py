import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

from datetime import datetime
from copy import deepcopy
import wandb
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.av_policy import RL_brain
from utils.environment import Env
from utils.function import set_random_seed, evaluate, train_av_online, train_av_offline, cm_result


batch_size = 128
train_size = 128
episodes = 10
learning_rate = 1e-4
alpha = 0.1
reward_type = 'r3'
auto_alpha = False
use_wandb = False
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42
name = datetime.now().strftime('%Y%m%d-%H%M')+'-test_av-seed='+str(random_seed) # for example: '20230509-1544-test_av-seed=42'
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all.npz')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, 
          reward_type=reward_type, seed=random_seed)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate, alpha=alpha)

if use_wandb:
    wandb_config = {
        'batch_size': batch_size, 
        'train_size': train_size, 
        'episodes': episodes, 
        'learning_rate': learning_rate, 
        'alpha': alpha, 
        'reward_type': reward_type, 
        'auto_alpha': auto_alpha, 
        'seed': random_seed, 
    }
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=name, 
        # entity='xyz_thu',
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

metrics_before, gt_label_before = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_before = 1 - np.sum(gt_label_before) / gt_label_before.size
print('Success rate before training: %.4f\n' % success_rate_before, metrics_before)

init_policynet_params = deepcopy(av_model.policy_net.state_dict())

index = lib.sample(size=train_size)
train_scenario = lib.data[index]
train_av_online(av_model, env, train_scenario, episodes, auto_alpha, wandb_logger)
# train_av_offline(av_model, env, train_scenario, epochs, episodes, auto_alpha, wandb_logger)

metrics_end, gt_label_end = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_end = 1 - np.sum(gt_label_end) / gt_label_end.size
print('Success rate after training: %.4f\n' % success_rate_end, metrics_end)
cm_result(gt_label_before, gt_label_end)

av_model.policy_net.load_state_dict(init_policynet_params)
metrics_best, gt_label_best = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_best = 1 - np.sum(gt_label_best) / gt_label_best.size
print('Success rate using loaded params: %.4f\n' % success_rate_best, metrics_best)
cm_result(gt_label_before, gt_label_best)

env.close()
