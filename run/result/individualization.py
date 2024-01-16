import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

from tqdm import trange
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_mlp
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import set_random_seed, evaluate, train_predictor


eval_size = 4096
train_size = 4096
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 92    # 14, 42, 51, 71, 92
name = './model/20230906-2231-fixed_alpha=0.1-seed='+str(random_seed)+'/round10_policy.pth'
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all.npz')
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, seed=random_seed)
av_model = RL_brain(env, device=device)
av_model.policy_net.load_state_dict(torch.load(name, map_location=device))
index = lib.sample(eval_size)
X_train = lib.data[index]


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
            # unable to perceive the vehicle ahead on the left
            state = np.delete(state, i, axis=0)
            state = np.vstack((state, np.zeros(4)))
    return state.reshape(-1)

def scenario_statistics(scenarios: np.ndarray) -> tuple[float, list[float]]:
    bv_num, bv_num_interest, bv_interest_dis, bv_av_pos_x, bv_av_pos_y = 0, 0, [], [], []
    for i in range(scenarios.shape[0]):
        scenario = scenarios[i].reshape(-1, 6)
        init_bv_state = scenario[(scenario[:, 0] == 0) & (scenario[:, 1] != 0)]
        for j in range(init_bv_state.shape[0]):
            bv_av_pos_x.append(init_bv_state[j, 2] - scenario[0, 2])
            bv_av_pos_y.append(init_bv_state[j, 3] - scenario[0, 3])
            if (init_bv_state[j, 2] > scenario[0, 2]) and (init_bv_state[j, 3] > scenario[0, 3]):
                # BV located on the left front of AV
                bv_num_interest += 1
                bv_interest_dis.append(np.sqrt((init_bv_state[j, 2] - scenario[0, 2]) ** 2 + 
                                      (init_bv_state[j, 3] - scenario[0, 3]) ** 2))
        bv_num += (j + 1)
    
    bv_interest_proportion = bv_num_interest / bv_num
    return bv_interest_proportion, bv_interest_dis, bv_av_pos_x, bv_av_pos_y


predictor_defect = predictor_mlp(input_dim=lib.max_dim, device=device)
predictor_defect.to(device)
y_train_defect = defect_evaluate(av_model, env, scenarios=X_train)
success_rate_defect = 1 - np.sum(y_train_defect) / eval_size
print('Success rate with defect:', success_rate_defect)
train_predictor(predictor_defect, X_train, y_train_defect, device=device)
lib.labeling(predictor_defect)
index = lib.select(size=train_size)
train_scenario_defect = lib.data[index]
bv_interest_proportion_defect, bv_dis_defect, bv_av_pos_x_defect, bv_av_pos_y_defect \
                = scenario_statistics(train_scenario_defect)
print('The proportion of BVs of interest:', bv_interest_proportion_defect)

predictor = predictor_mlp(input_dim=lib.max_dim, device=device)
predictor.to(device)
_, y_train = evaluate(av_model, env, scenarios=X_train)
success_rate = 1 - np.sum(y_train) / eval_size
print('Success rate without defect:', success_rate)
train_predictor(predictor, X_train, y_train, device=device)
lib.labeling(predictor)
index = lib.select(size=train_size)
train_scenario = lib.data[index]
bv_interest_proportion, bv_dis, bv_av_pos_x, bv_av_pos_y = scenario_statistics(train_scenario)
print('The proportion of BVs of interest:', bv_interest_proportion)

bin_edges = np.linspace(0, 80, 21)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
hist_defect, _ = np.histogram(bv_dis_defect, bins=bin_edges, density=True)
hist, _ = np.histogram(bv_dis, bins=bin_edges, density=True)
plt.figure(figsize=(6, 3))
plt.bar(bin_centers, hist_defect, width=4, alpha=0.5, label='with defect')
plt.bar(bin_centers, hist, width=4, alpha=0.5, label='without defect')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('./figure/individualization_bv_av_dis.pdf', bbox_inches='tight')

index = random.sample(range(len(bv_dis)), 1000)
bv_av_pos_x = [bv_av_pos_x[i] for i in index]
bv_av_pos_y = [bv_av_pos_y[i] for i in index]
bv_av_pos_x_defect = [bv_av_pos_x_defect[i] for i in index]
bv_av_pos_y_defect = [bv_av_pos_y_defect[i] for i in index]
plt.figure(figsize=(9, 2))
plt.scatter(np.array(bv_av_pos_x, dtype=np.float16), np.array(bv_av_pos_y, dtype=np.float16),
            s=2, alpha=0.4, label='without defect')
plt.scatter(np.array(bv_av_pos_x_defect, dtype=np.float16), np.array(bv_av_pos_y_defect, dtype=np.float16),
            s=2, alpha=0.4, label='with defect')
plt.scatter(0, 0, s=20, c='r', label='AV')
plt.grid()
plt.legend()
plt.axis('equal')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.savefig('./figure/individualization_bv_av_pos.pdf', bbox_inches='tight')

env.close()
