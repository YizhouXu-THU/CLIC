import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import time
from datetime import datetime
import wandb
import numpy as np
import torch
import random
from copy import deepcopy

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import (set_random_seed, train_predictor, evaluate, 
                            train_av_offline, train_av_online, cm_result, matrix_test, draw_surface)


# 0. Prepare
t0 = time.time()

eval_size = 4096
batch_size = 128
train_size = 128
rounds = 10
epochs = 20
episodes = 10
learning_rate = 1e-4
alpha = 0.1
reward_type = 'r3'
auto_alpha = True
use_wandb = True
sumo_gui = False
save_model = True
all_fail = False
KEYFRAME = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42    # 14, 42, 51, 71, 92
if all_fail:
    name = datetime.now().strftime('%Y%m%d-%H%M')+'-fail-seed='+str(random_seed)  # for example: '20230509-1544-fail-seed=42'
else:
    name = datetime.now().strftime('%Y%m%d-%H%M')+'-sample_fail-seed='+str(random_seed)  # for example: '20230509-1544-sample_fail-seed=42'
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all.npz')
predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
predictor.to(device)
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, 
          reward_type=reward_type, seed=random_seed)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate, alpha=alpha)
policy_net_params = []
eval_success_rate = np.zeros(rounds)

if save_model:
    save_path = './model/' + name + '/'
    os.makedirs(save_path, exist_ok=True)
    torch.save(av_model.policy_net.state_dict(), save_path+'round0_policy.pth')

if use_wandb:
    wandb_config = {
        'eval_size': eval_size, 
        'batch_size': batch_size, 
        'train_size': train_size,
        'rounds': rounds, 
        'epochs': epochs, 
        'episodes': episodes, 
        'learning_rate': learning_rate, 
        'auto_alpha': auto_alpha, 
        'seed': random_seed, 
    }
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing',
        name=name,
        entity='xyz_thu',
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

t1 = time.time()
print('Preparation time: %.1fs' % (t1-t0))

# pretrain
# print('Pretraining')
# index = lib.sample(size=5*train_size)
# train_scenario = lib.data[index]
# av_model = train_av_online(av_model, env, train_scenario, episodes, auto_alpha)
# t2 = time.time()
# print('Pretraining time: %.1fs' % (t2-t1))
# index = lib.sample(size=eval_size)
# X_train = lib.data[index]
# _, y_train = evaluate(av_model, env, scenarios=X_train)
# success_rate = 1 - np.sum(y_train) / eval_size
# if use_wandb:
#     wandb_logger.log({'Evaluate success rate': success_rate})
# print('Evaluate success rate: %.3f' % success_rate)
# t3 = time.time()
# print('Evaluation time: %.1fs' % (t3-t2))

# test on all scenarios
print('    Evaluating')
metrics_before, gt_label_before = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(gt_label_before) / gt_label_before.size
print('Success rate after pretraining: %.4f\n' % success_rate, metrics_before)
t2 = time.time()
print('Evaluation time: %.1fs' % (t2-t1))

for i in range(rounds):
    print('Round %d' % (round+1))
    index = lib.sample(size=10*batch_size)
    test_scenario = lib.data[index]
    scenario_num = test_scenario.shape[0]
    fail_count = 0
    print('    Collecting failed scenarios')
    
    # find failed scenarios and put them into batch
    for j in range(scenario_num):
        state = env.reset(test_scenario[j])
        done = False
        step = 0
        k = KEYFRAME
        while not done:
            step += 1
            if k == KEYFRAME:
                action = av_model.choose_action(state, deterministic=False)
                pre_action = action
                k = 1
            else:
                action = pre_action
                k += 1
            next_state, reward, done, info = env.step(action, timestep=step)
            state = next_state
        
        if info == 'fail':
            fail_count += 1
            if fail_count == 1:
                batch = [test_scenario[j]]
                batch = np.array(batch)
            else:
                batch = np.append(batch, [test_scenario[j]], axis=0)
            if all_fail:
                # randomly choose a scenario to put into batch
                batch = np.append(batch, [test_scenario[random.randint(0, scenario_num-1)]], axis=0)
            if batch.shape[0] >= batch_size:
                print('    Failed scenarios collected')
                break
    
    # train on failed scenarios
    t4 = time.time()
    print('    Training')
    train_av_online(av_model, env, batch, episodes, auto_alpha, wandb_logger)
    policy_net_params.append(deepcopy(av_model.policy_net.state_dict()))
    if save_model:
        torch.save(av_model.policy_net.state_dict(), './model/'+name+'/round'+str(i+1)+'_policy.pth')
    t5 = time.time()
    print('    Training time: %.1fs' % (t5-t4))

    # evaluate
    index = lib.sample(size=eval_size)
    X_train = lib.data[index]
    _, y_train = evaluate(av_model, env, scenarios=X_train)
    success_rate = 1 - np.sum(y_train) / eval_size
    eval_success_rate[i] = success_rate
    if use_wandb:
        wandb_logger.log({'Evaluate success rate': success_rate})
    print('    Evaluate success rate: %.3f' % success_rate)
    t6 = time.time()
    print('    Evaluation time: %.1fs' % (t6-t5))

# test on all scenarios
t6 = time.time()

metrics_end, gt_label_end = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(gt_label_end) / gt_label_end.size
print('Success rate after training: %.4f\n' % success_rate, metrics_end)
cm_result(gt_label_before, gt_label_end)

best_round_index = np.argmax(eval_success_rate)
av_model.policy_net.load_state_dict(policy_net_params[best_round_index])
metrixs_best, gt_label_best = evaluate(av_model, env, scenarios=lib.data)
success_rate_best = 1 - np.sum(gt_label_best) / gt_label_best.size
print('The best round: %d, Success rate of the best round: %.4f' % (best_round_index+1, success_rate_best))
cm_result(gt_label_before, gt_label_best)

t7 = time.time()
print('Evaluation time: %.1fs' % (t7-t6))
print('Total time: %.1fs' % (t7-t0))

env.close()
