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


sumo_gui = True
delay = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 92    # 14, 42, 51, 71, 92
name_CL = '20230906-2231-fixed_alpha=0.1-seed=' + str(random_seed)
name_simple = '20230819-0150-simple-seed=' + str(random_seed)
name_individualization = '20230920-0054-individualization-seed=' + str(random_seed)
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
predictor.to(device)
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, delay=delay, seed=random_seed)
av_model = RL_brain(env, capacity=0, device=device)

log = open('./log/video.log', 'w')
sys.stdout = log

# baseline experiment
count = 0
indices_baseline = []
for i in range(10000):
    index = lib.sample(size=1)
    scenario = lib.data[index]
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth', map_location=device))
    _, label_before = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_simple + '/round10_policy.pth', map_location=device))
    _, label_simple = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth', map_location=device))
    _, label_CL = evaluate(av_model, env, scenario)
    
    if (label_before[0] == 1) and (label_simple[0] == 1) and (label_CL[0] == 0):
        indices_baseline.append(index[0])
        count += 1
    
    if count == 10:
        print('Baseline experiment:', indices_baseline)
        break

# matrix experiment
# fix predictor = 5
count = 0
indices_matrix_pred = []
for i in range(10000):
    predictor.load_state_dict(torch.load('./model/' + name_CL + '/round5_predictor.pth', map_location=device))
    lib.labeling(predictor)
    index_pred_5 = lib.select(size=1)
    scenario = lib.data[index_pred_5]
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth', map_location=device))
    _, label_1_5 = evaluate(av_model, env, scenario)
    
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth', map_location=device))
    _, label_10_5 = evaluate(av_model, env, scenario)
    
    if (label_1_5[0] == 1) and (label_10_5[0] == 0):
        indices_matrix_pred.append(index_pred_5[0])
        count += 1
    
    if count == 10:
        print('Matrix experiment: Fixed predictor:', indices_matrix_pred)
        break

# fix AV = 5
count = 0
indices_matrix_av = []
for i in range(10000):
    av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round5_policy.pth', map_location=device))
    
    predictor.load_state_dict(torch.load('./model/' + name_CL + '/round1_predictor.pth', map_location=device))
    lib.labeling(predictor)
    index_5_1 = lib.select(size=1)
    scenario = lib.data[index_5_1]
    _, label_5_1 = evaluate(av_model, env, scenario)
    
    predictor.load_state_dict(torch.load('./model/' + name_CL + '/round10_predictor.pth', map_location=device))
    lib.labeling(predictor)
    index_5_10 = lib.select(size=1)
    scenario = lib.data[index_5_10]
    _, label_5_10 = evaluate(av_model, env, scenario)
    
    if (label_5_1[0] == 0) and (label_5_10[0] == 1):
        indices_matrix_av.append([index_5_1[0], index_5_10[0]])
        count += 1
    
    if count == 10:
        print('Matrix experiment: Fixed AV:', indices_matrix_av)
        break

# individualization experiment
count = 0
indices_normal = []
predictor.load_state_dict(torch.load('./model/' + name_individualization + '/predictor.pth', map_location=device))
lib.labeling(predictor)
# for i in range(10000):
#     index = lib.select(size=1)
#     scenario = lib.data[index].reshape(-1, 6)
#     init_bv_state = scenario[(scenario[:, 0] == 0) & (scenario[:, 1] != 0)]
#     for j in range(init_bv_state.shape[0]):
#         if (init_bv_state[j, 2] > scenario[0, 2]) and (init_bv_state[j, 3] > scenario[0, 3]):
#             # BV located on the left front of AV
#             indices_normal.append(index[0])
#             count += 1
#             break
    
#     if count == 10:
#         print('Individualization experiment: Normal:', indices_normal)
#         break
indices_normal = lib.select(size=10).tolist()
print('Individualization experiment: Normal:', indices_normal)

count = 0
indices_defect = []
predictor.load_state_dict(torch.load('./model/' + name_individualization + '/predictor_defect.pth', map_location=device))
lib.labeling(predictor)
for i in range(10000):
    index = lib.select(size=1)
    scenario = lib.data[index].reshape(-1, 6)
    init_bv_state = scenario[(scenario[:, 0] == 0) & (scenario[:, 1] != 0)]
    for j in range(init_bv_state.shape[0]):
        if (init_bv_state[j, 2] > scenario[0, 2]) and (init_bv_state[j, 3] > scenario[0, 3]):
            # BV located on the left front of AV
            indices_defect.append(index[0])
            count += 1
            break
    
    if count == 10:
        print('Individualization experiment: Defect:', indices_defect)
        break


def defect_evaluate(av_model, env, scenarios: np.ndarray) -> np.ndarray:
    scenario_num = scenarios.shape[0]
    labels = np.zeros(scenario_num)
    
    with torch.no_grad():
        # for i in trange(scenario_num):
        for i in range(scenario_num):
            state = env.reset(scenarios[i])
            done = False
            step = 0
            while not done:
                step += 1
                action = av_model.choose_action(process_state(state), deterministic=True)
                next_state, reward, done, info = env.step(action, timestep=step, need_reward=False)
                state = next_state
            if info == 'fail':
                labels[i] = 1
            elif info == 'succeed':
                labels[i] = 0
    
    return labels

def process_state(state: np.ndarray) -> np.ndarray:
    state = state.reshape(-1, 4)
    for i in range(1, state.shape[0]):
        if (state[i, 0] > 0) and (state[i, 1] > 0):
            state[i, 0], state[i, 1] = 100, 100     # unable to perceive the vehicle ahead on the left
    return state.reshape(-1)


# # indices_baseline = np.array([ 5348, 18407, 13299, 24498, 61668, 61674, 27400, 17003, 31751, 64172])
indices_baseline = np.array([47867, 32157, 48248, 42768, 1183, 11322, 617, 17014, 37686, 37035])
indices_matrix_pred = np.array([11430, 41948, 51227, 21068, 47176, 28651, 46525, 57168, 30522, 59252])
indices_matrix_av = np.array([[31661, 38216], [34234, 50914], [ 3687, 59983], [13135, 59725], [10113, 64836], 
                              [44129, 32468], [21045, 65391], [39480, 65335], [17070, 47710], [54230, 62626]])
# indices_normal = np.array([40982, 16731, 55422, 29294, 36521, 20184, 20641, 32303, 54033, 12243])
indices_normal = np.array([45151, 30095, 22871, 10797, 48639, 65407, 24735, 58743, 10636, 26395])
indices_defect = np.array([54778, 58253, 36990, 50276, 38593, 17197, 10661, 48924, 19809, 11544])

# baseline experiment
av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth', map_location=device))
scenarios = lib.data[indices_baseline]
evaluate(av_model, env, scenarios)

av_model.policy_net.load_state_dict(torch.load('./model/' + name_simple + '/round10_policy.pth', map_location=device))
scenarios = lib.data[indices_baseline]
evaluate(av_model, env, scenarios)

av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth', map_location=device))
scenarios = lib.data[indices_baseline]
evaluate(av_model, env, scenarios)

# matrix experiment
# fix predictor = 5
scenarios_5 = lib.data[indices_matrix_pred]

av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round1_policy.pth', map_location=device))
evaluate(av_model, env, scenarios_5)

av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth', map_location=device))
evaluate(av_model, env, scenarios_5)

# fix AV = 5
av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round5_policy.pth', map_location=device))
scenarios_1 = lib.data[indices_matrix_av[:, 0]]
scenarios_10 = lib.data[indices_matrix_av[:, 1]]
evaluate(av_model, env, scenarios_1)
evaluate(av_model, env, scenarios_10)

# # individualization experiment
av_model.policy_net.load_state_dict(torch.load('./model/' + name_CL + '/round10_policy.pth', map_location=device))
evaluate(av_model, env, lib.data[indices_normal])
defect_evaluate(av_model, env, lib.data[indices_defect])

env.close()
