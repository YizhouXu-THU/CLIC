import time
from datetime import datetime
import numpy as np
import wandb

from utils.scenario_lib import scenario_lib
from utils.reward_predictor import reward_predictor
from utils.env import Env
from utils.av_policy import SAC
from utils.func import train_predictor, evaluate, train_av


def main():
    # 0. Prepare
    t0 = time.time()
    
    eval_size = 4096
    train_size = 100
    rounds = 10
    epochs = 500
    episodes = 100
    use_wandb = False
    sumo_gui = True
    
    lib = scenario_lib(path='./scenario_lib_test/')
    predictor = reward_predictor(num_input=lib.max_dim)
    env = Env(max_bv_num=lib.max_bv_num, gui=sumo_gui)
    av_model = SAC(env)
    
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
        print('Round %d' % round)
        t1 = time.time()
        
        # 1. Sample
        index = lib.sample(size=eval_size)
        X_train = lib.data[index]
        t2 = time.time()
        print('    Sampling time: %.1fs' % (t2-t1))

        # 2. Evaluate (Interact)
        y_train = evaluate(av_model, env, scenarios=X_train, size=eval_size)
        success_rate = 1 - np.sum(y_train) / eval_size
        if use_wandb:
            wandb_logger.log({'Success rate': success_rate})
        print('Success rate: %.3f' % success_rate)
        t3 = time.time()
        print('    Evaluation time: %.1fs' % (t3-t2))

        # 3. Train reward predictor
        predictor = train_predictor(predictor, X_train, y_train, epochs=epochs, wandb_logger=wandb_logger)
        t4 = time.time()
        print('    Training reward predictor time: %.1fs' % (t4-t3))

        # 4. Labeling
        lib.labeling(predictor)
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


if __name__ == '__main__':
    main()
