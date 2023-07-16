import os
import sys
sys.path.append(os.getcwd())

import time
from datetime import datetime
import wandb
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor
from utils.environment import Env
from utils.av_policy import RL_brain
from utils.function import train_predictor, evaluate, train_av


# 0. Prepare
t0 = time.time()

eval_size = 4096
batch_size = 128
train_size = 128
rounds = 10
epochs = 20
episodes = 20
learning_rate = 1e-4
use_wandb = True
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

lib = scenario_lib(path='./scenario_lib/', npy_path='./all_data.npy')
pred = predictor(num_input=lib.max_dim, device=device)
pred.to(device)
env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', gui=sumo_gui)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate)

if use_wandb:
    wandb_config = {
        'eval_size': eval_size, 
        'batch_size': batch_size, 
        'train_size': train_size,
        'rounds': rounds, 
        'epochs': epochs, 
        'episodes': episodes, 
        'learning_rate': learning_rate, 
    }
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=datetime.now().strftime('%Y%m%d-%H%M')+'-CL',  # for example: '20230509-1544-CL'
        config=wandb_config, 
        reinit=True, 
        )
else:
    wandb_logger = None

# TODO: may need to pretrain av_model
t1 = time.time()
print('Preparation time: %.1fs' % (t1-t0))

# test on all scenarios
all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate before training: %.3f' % success_rate)
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
    y_train = evaluate(av_model, env, scenarios=X_train)
    success_rate = 1 - np.sum(y_train) / eval_size
    if use_wandb:
        wandb_logger.log({'Evaluate success rate': success_rate})
    print('    Evaluate success rate: %.3f' % success_rate)
    t4 = time.time()
    print('    Evaluation time: %.1fs' % (t4-t3))

    # 3. Train reward predictor
    pred = train_predictor(pred, X_train, y_train, epochs=epochs, lr=learning_rate, 
                            batch_size=batch_size, wandb_logger=wandb_logger, device=device)
    t5 = time.time()
    print('    Training reward predictor time: %.1fs' % (t5-t4))

    # 4. Labeling
    lib.labeling(pred)
    t6 = time.time()
    print('    Labeling time: %.1fs' % (t6-t5))

    # 5. Select
    index = lib.select(size=train_size)
    train_scenario = lib.data[index]
    t7 = time.time()
    print('    Selecting time: %.1fs' % (t7-t6))

    # 6. Train AV model
    av_model = train_av(av_model, env, scenarios=train_scenario, 
                        epochs=epochs, episodes=episodes, wandb_logger=wandb_logger)
    t8 = time.time()
    print('    Training AV model time: %.1fs' % (t8-t7))
    
    if use_wandb:
        wandb_logger.log({'Round time': t8-t2, 'Total time': t8-t0})
    print('    Time of the whole round: %.1fs' % (t8-t2))
    print('Total time: %.1fs' % (t8-t0))

# test on all scenarios
t8 = time.time()
all_label = evaluate(av_model, env, scenarios=lib.data)
success_rate = 1 - np.sum(all_label) / all_label.size
print('Success rate after training: %.3f' % success_rate)
t9 = time.time()
print('Evaluation time: %.1fs' % (t9-t8))
print('Total time: %.1fs' % (t9-t0))

env.close()
