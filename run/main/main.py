import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import time
from datetime import datetime
from copy import deepcopy
import wandb
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn
from utils.environment import Env
from utils.av_policy import RL_brain
# from utils.av_policy_per import RL_brain
from utils.function import (set_random_seed, train_predictor, evaluate, 
                            train_av_online, cm_result, matrix_test, draw_surface)


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
auto_alpha = False
use_wandb = True
sumo_gui = False
save_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42    # 14, 42, 51, 71, 92
name = datetime.now().strftime('%Y%m%d-%H%M')+'-CLIC-seed='+str(random_seed)    # for example: '20230509-1544-CLIC-seed=42'
set_random_seed(random_seed)

lib = scenario_lib(path='./data/all.npz')
predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
predictor.to(device)
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui, 
          reward_type=reward_type, seed=random_seed)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate, alpha=alpha)
predictor_params = []
policy_net_params = [deepcopy(av_model.policy_net.state_dict())]
if save_model:
    save_path = './model/' + name + '/'
    os.makedirs(save_path, exist_ok=True)
    torch.save(av_model.policy_net.state_dict(), save_path+'round0_policy.pth')

eval_success_rate = np.zeros(rounds)

if use_wandb:
    wandb_config = {
        'eval_size': eval_size, 
        'batch_size': batch_size, 
        'train_size': train_size,
        'rounds': rounds, 
        'epochs': epochs, 
        'episodes': episodes, 
        'learning_rate': learning_rate, 
        'alpha': alpha, 
        'reward_type': reward_type, 
        'auto_alpha': auto_alpha, 
        'seed': random_seed, 
        }
    wandb_logger = wandb.init(
        project='CLIC', 
        name=name, 
        # entity='xyz_thu',
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

t1 = time.time()
print('Preparation time: %.1fs' % (t1-t0))

# test on all scenarios
metrics_before, gt_label_before = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_before = 1 - np.sum(gt_label_before) / gt_label_before.size
print('Success rate before training: %.4f\n' % success_rate_before, metrics_before)
t2 = time.time()
print('Evaluation time: %.1fs' % (t2-t1))

# main loop
for round in range(rounds):
    if use_wandb:
        wandb_logger.log({'Round': round})
    print('Round %d' % (round+1))
    t2 = time.time()
    
    # 1. Sample
    index = lib.sample(size=eval_size)
    X_train = lib.data[index]
    t3 = time.time()
    print('    Sampling time: %.1fs' % (t3-t2))

    # 2. Evaluate (Interact)
    _, y_train = evaluate(av_model, env, scenarios=X_train)
    success_rate = 1 - np.sum(y_train) / eval_size
    eval_success_rate[round] = success_rate
    if use_wandb:
        wandb_logger.log({'Evaluate success rate': success_rate})
    print('    Evaluate success rate: %.4f' % success_rate)
    t4 = time.time()
    print('    Evaluation time: %.1fs' % (t4-t3))

    # 3. Train reward predictor
    train_predictor(predictor, X_train, y_train, epochs=epochs, lr=learning_rate, 
                    batch_size=batch_size, wandb_logger=None, device=device)
    predictor_params.append(deepcopy(predictor.state_dict()))
    if save_model:
        torch.save(predictor.state_dict(), './model/'+name+'/round'+str(round+1)+'_predictor.pth')
    t5 = time.time()
    print('    Training reward predictor time: %.1fs' % (t5-t4))

    # 4. Labeling
    lib.labeling(predictor)
    t6 = time.time()
    print('    Labeling time: %.1fs' % (t6-t5))

    # 5. Select
    index = lib.select(size=train_size)
    train_scenario = lib.data[index]
    t7 = time.time()
    print('    Selecting time: %.1fs' % (t7-t6))

    # 6. Train AV model
    train_av_online(av_model, env, train_scenario, episodes, auto_alpha, wandb_logger)
    policy_net_params.append(deepcopy(av_model.policy_net.state_dict()))
    if save_model:
        torch.save(av_model.policy_net.state_dict(), './model/'+name+'/round'+str(round+1)+'_policy.pth')
    t8 = time.time()
    print('    Training AV model time: %.1fs' % (t8-t7))
    
    if use_wandb:
        wandb_logger.log({'Round time': t8-t2, 'Total time': t8-t0})
    print('    Time of the whole round: %.1fs' % (t8-t2))
    print('Total time: %.1fs' % (t8-t0))

# test on all scenarios
t8 = time.time()

metrics_end, gt_label_end = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_end = 1 - np.sum(gt_label_end) / gt_label_end.size
print('Success rate after training: %.4f\n' % success_rate_end, metrics_end)
cm_result(gt_label_before, gt_label_end)

best_round_index = np.argmax(eval_success_rate)
av_model.policy_net.load_state_dict(policy_net_params[best_round_index])
metrics_best, gt_label_best = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
success_rate_best = 1 - np.sum(gt_label_best) / gt_label_best.size
print('The best round: %d, Success rate of the best round: %.4f\n' % (best_round_index+1, success_rate_best), metrics_best)
cm_result(gt_label_before, gt_label_best)

t9 = time.time()
print('Evaluation time: %.1fs' % (t9-t8))

# matrix test
# matrix_results = matrix_test(predictor_params, policy_net_params, av_model, predictor, env, lib, eval_size, device)
# np.set_printoptions(precision=4)
# print('Matrix experiment results:\n', matrix_results)
# draw 3D surface graph
# draw_surface(matrix_results, filename='3D_matrix.png')

t10 = time.time()
print('Matrix experiment time: %.1fs' % (t10-t9))
print('Total time: %.1fs' % (t10-t0))

env.close()
