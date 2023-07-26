import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import evaluate, evaluate_random


# Prepare
sumo_gui = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

all_lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
crash_lib = scenario_lib(path='./data/crash/', npy_path='./data/crash.npy')
env = Env(max_bv_num=all_lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=0, device=device)

all_label_rl = evaluate(av_model, env, scenarios=all_lib.data)
crash_label_rl = evaluate(av_model, env, scenarios=crash_lib.data)

all_label_random = evaluate_random(env, scenarios=all_lib.data)
crash_label_random = evaluate_random(env, scenarios=crash_lib.data)

all_success_rate_rl = 1 - np.sum(all_label_rl) / all_label_rl.size
crash_success_rate_rl = 1 - np.sum(crash_label_rl) / crash_label_rl.size
print('Using untrained RL:')
print('All scenario success rate: %.3f;' % all_success_rate_rl, 
      'Crash scenario success rate: %.3f.' % crash_success_rate_rl)

all_success_rate_random = 1 - np.sum(all_label_random) / all_label_random.size
crash_success_rate_random = 1 - np.sum(crash_label_random) / crash_label_random.size
print('Using random policy:')
print('All scenario success rate: %.3f;' % all_success_rate_random, 
      'Crash scenario success rate: %.3f.' % crash_success_rate_random)
