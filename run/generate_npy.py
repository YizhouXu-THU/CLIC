import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import evaluate


# Prepare
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

lib = scenario_lib(path='./data/all/', npy_path='')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=0, device=device)

all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate: %.3f' % success_rate)

all_data = np.append(lib.data, all_label.reshape(-1,1), axis=1)
np.save('./data/all.npy', all_data)
env.close()
