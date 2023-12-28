import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import set_random_seed, evaluate


sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42
set_random_seed(random_seed)

lib = scenario_lib(path='./data/example/')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=0, device=device)

_, gt_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(gt_label) / gt_label.size
print('Success rate: %.4f' % success_rate)

scenario_data = np.append(lib.data, gt_label.reshape(-1,1), axis=1)
# np.save('./data/example.npy', scenario_data)
np.savez('./data/example.npz', data=scenario_data, type_count=lib.type_count, max_bv_num=lib.max_bv_num)

env.close()
