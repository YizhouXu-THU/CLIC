import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

from datetime import datetime
import wandb
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import set_random_seed, evaluate


use_wandb = True
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42  # 14, 42, 51, 71, 92
name = datetime.now().strftime('%Y%m%d-%H%M')+'-test_SR-seed='+str(random_seed)     # for example: '20230509-1544-test_SR-seed=42'
set_random_seed(random_seed)

all_lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
crash_lib = scenario_lib(path='./data/crash/', npy_path='./data/crash.npy')
env = Env(max_bv_num=all_lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=0, device=device)
if use_wandb:
    wandb_config = {'seed': random_seed}
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=name, 
        # entity='xyz_thu',
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

# untrained RL
print('Using untrained RL:')
all_metrics_rl, all_label_rl = evaluate(av_model, env, scenarios=all_lib.data, need_metrics=True)
all_success_rate_rl = 1 - np.sum(all_label_rl) / all_label_rl.size
print('All scenario success rate: %.4f\n' % all_success_rate_rl, all_metrics_rl)

crash_metrics_rl, crash_label_rl = evaluate(av_model, env, scenarios=crash_lib.data, need_metrics=True)
crash_success_rate_rl = 1 - np.sum(crash_label_rl) / crash_label_rl.size
print('Crash scenario success rate: %.4f\n' % crash_success_rate_rl, crash_metrics_rl)

# random policy
print('Using random policy:')
all_metrics_random, all_label_random = evaluate(av_model=None, env=env, scenarios=all_lib.data, need_metrics=True)
all_success_rate_random = 1 - np.sum(all_label_random) / all_label_random.size
print('All scenario success rate: %.4f\n' % all_success_rate_random, all_metrics_random)

crash_metrics_random, crash_label_random = evaluate(av_model=None, env=env, scenarios=crash_lib.data, need_metrics=True)
crash_success_rate_random = 1 - np.sum(crash_label_random) / crash_label_random.size
print('Crash scenario success rate: %.4f\n' % crash_success_rate_random, crash_metrics_random)
