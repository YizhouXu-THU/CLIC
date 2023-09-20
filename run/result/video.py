import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import set_random_seed, evaluate

log = open('./log/video.log', 'w')
sys.stdout = log


sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42    # 14, 42, 51, 71, 92
name_CL = '20230906-2231-fixed_alpha=0.1-seed=' + str(random_seed)
name_simple = '20230819-0150-simple-seed=' + str(random_seed)
name_individualization = '20230920-0054-individualization-seed=' + str(random_seed)
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
predictor.to(device)
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, seed=random_seed)
av_model = RL_brain(env, capacity=0, device=device)

# baseline
count = 0
indices_baseline = []
for i in range(10000):
    index = lib.sample(size=1)
    scenario = lib.data[index]
    _, label_before = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_simple + '/round10_policy.pth'))
    _, label_simple = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth'))
    _, label_CL = evaluate(av_model, env, scenario)
    
    if (label_before[0] == 1) and (label_simple[0] == 1) and (label_CL[0] == 0):
        indices_baseline.append(index)
        count += 1
        print(count, end=' ')
    
    if count == 10:
        print('\nBaseline experiment:', indices_baseline)
        break

# matrix
count = 0
indices_matrix = []

for i in range(10000):
    index = lib.sample(size=1)
    scenario = lib.data[index]
    
    predictor.load_state_dict(torch.load('./model/' + name_CL + '/round1_predictor.pth'))
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth'))
    _, label_1_1 = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth'))
    _, label_10_1 = evaluate(av_model, env, scenario)
    
    predictor.load_state_dict(torch.load('./model/' + name_CL + '/round10_predictor.pth'))
    _, label_10_10 = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth'))
    _, label_1_10 = evaluate(av_model, env, scenario)
    
    if (label_10_1[0] == 0) and (label_1_10[0] == 1):
        indices_matrix.append(index)
        count += 1
        print(count, end=' ')
    
    if count == 10:
        print('\nMatrix experiment:', indices_matrix)
        break

# individualization
count = 0
indices_normal = []
predictor.load_state_dict(torch.load('./model/' + name_individualization + '/predictor.pth'))
for i in range(10000):
    index = lib.sample(size=1)
    scenario = lib.data[index].reshape(-1, 6)
    init_bv_state = scenario[(scenario[:, 0] == 0) & (scenario[:, 1] != 0)]
    for j in range(init_bv_state.shape[0]):
        if (init_bv_state[j, 2] > scenario[0, 2]) and (init_bv_state[j, 3] > scenario[0, 3]):
            # BV located on the left front of AV
            indices_normal.append(index)
            count += 1
            print(count, end=' ')
            break
    
    if count == 10:
        print('\nIndividualization experiment: Normal: ', indices_normal)
        break

count = 0
indices_defect = []
predictor.load_state_dict(torch.load('./model/' + name_individualization + '/predictor_defect.pth'))
for i in range(10000):
    index = lib.sample(size=1)
    scenario = lib.data[index].reshape(-1, 6)
    init_bv_state = scenario[(scenario[:, 0] == 0) & (scenario[:, 1] != 0)]
    for j in range(init_bv_state.shape[0]):
        if (init_bv_state[j, 2] > scenario[0, 2]) and (init_bv_state[j, 3] > scenario[0, 3]):
            # BV located on the left front of AV
            indices_defect.append(index)
            count += 1
            print(count, end=' ')
            break
    
    if count == 10:
        print('\nIndividualization experiment: Defect: ', indices_defect)
        break

env.close()
