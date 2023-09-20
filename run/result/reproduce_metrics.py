import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import wandb
import torch

from utils.scenario_lib import scenario_lib
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import set_random_seed, evaluate


use_wandb = True
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42    # 14, 42, 51, 71, 92
name = '20230819-0150-CL-seed=' + str(random_seed)
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, seed=random_seed)
av_model = RL_brain(env, capacity=0, device=device)
av_model.policy_net.load_state_dict(torch.load('./model/'+name+'/round10_policy.pth'))

if use_wandb:
    wandb_config = {'seed': random_seed}
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=name+'-metrics', 
        entity='xyz_thu',
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

metrics, labels = evaluate(av_model, env, lib.data, need_metrics=True)
print(metrics)

env.close()
