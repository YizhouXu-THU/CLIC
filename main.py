import time
from datetime import datetime
import numpy as np
import wandb
import torch

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor
from utils.environment import Env
from utils.av_policy import SAC
from utils.function import train_predictor, evaluate, train_av


def main():
    # 0. Prepare
    t0 = time.time()
    
    eval_size = 4096
    batch_size = 64
    train_size = 100
    rounds = 10
    epochs = 100
    episodes = 100
    learning_rate = 1e-4
    use_wandb = True
    sumo_gui = False
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    lib = scenario_lib(path='./scenario_lib/')
    pred = predictor(num_input=lib.max_dim, device=device)
    pred.to(device)
    env = Env(max_bv_num=lib.max_bv_num, gui=sumo_gui)
    av_model = SAC(env, device=device)
    
    if use_wandb:
        wandb_logger = wandb.init(
            project='CL for Autonomous Vehicle Training and Testing', 
            name=datetime.now().strftime('%Y%m%d-%H%M')+'-CL',  # for example: '20230509-1544-CL'
            reinit=True, 
            )
    else:
        wandb_logger = None
    
    # TODO: may need to pretrain av_model
    t1 = time.time()
    print('    Preparation time: %.1fs' % (t1-t0))

    # main loop
    for round in range(rounds):
        if use_wandb:
            wandb_logger.log({'Round': round})
        print('Round %d' % (round+1))
        t1 = time.time()
        
        # 1. Sample
        index = lib.sample(size=eval_size)
        X_train = lib.data[index]
        t2 = time.time()
        print('    Sampling time: %.1fs' % (t2-t1))

        # 2. Evaluate (Interact)
        y_train = evaluate(av_model, env, scenarios=X_train)
        success_rate = 1 - np.sum(y_train) / eval_size
        if use_wandb:
            wandb_logger.log({'Success rate': success_rate})
        print('    Success rate: %.3f' % success_rate)
        t3 = time.time()
        print('    Evaluation time: %.1fs' % (t3-t2))

        # 3. Train reward predictor
        pred = train_predictor(pred, X_train, y_train, epochs=epochs, lr=learning_rate, 
                               batch_size=batch_size, wandb_logger=wandb_logger, device=device)
        t4 = time.time()
        print('    Training reward predictor time: %.1fs' % (t4-t3))

        # 4. Labeling
        lib.labeling(pred)
        t5 = time.time()
        print('    Labeling time: %.1fs' % (t5-t4))

        # 5. Select
        index = lib.select(size=train_size)
        train_scenario = lib.data[index]
        t6 = time.time()
        print('    Selecting time: %.1fs' % (t6-t5))

        # 6. Train AV model
        av_model = train_av(av_model, env, scenarios=train_scenario, episodes=episodes, wandb_logger=wandb_logger)
        t7 = time.time()
        print('    Training AV model time: %.1fs' % (t7-t6))
        
        if use_wandb:
            wandb_logger.log({'Round time': t7-t1, 'Total time': t7-t0})
        print('    Time of the whole round: %.1fs' % (t7-t1))
        print('Total time: %.1fs\n' % (t7-t0))

    env.close()


if __name__ == '__main__':
    main()
