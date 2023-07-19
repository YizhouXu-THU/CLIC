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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

all_lib = scenario_lib(path='./scenario_lib/', npy_path='./all_data.npy')
crash_lib = scenario_lib(path='./scenario_lib_crash/', npy_path='./crash_data.npy')
env = Env(max_bv_num=all_lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=0, device=device)

all_label = evaluate(av_model, env, scenarios=all_lib.data)
crash_label = evaluate(av_model, env, scenarios=crash_lib.data)
all_success_rate = 1 - np.sum(all_label) / all_label.size
crash_success_rate = 1 - np.sum(crash_label) / crash_label.size
print('All scenario success rate: %.3f' % all_success_rate, 
      'Crash scenario success rate: %.3f' % crash_success_rate)
