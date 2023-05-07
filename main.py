import time
import numpy as np

from utils.scenario_lib import scenario_lib
from utils.reward_predictor import reward_predictor
from utils.env import Env
from utils.av_policy import SAC
from utils.func import train_predictor, evaluate, train_av


def main():
    t0 = time.time()
    lib = scenario_lib(path='./scenario_lib_test/')
    predictor = reward_predictor(num_input=lib.max_dim)
    env = Env(max_bv_num=lib.max_bv_num, gui=False)
    av_model = SAC(env)
    # TODO: may need to pretrain av_model
    t1 = time.time()
    print('Preparation time: %.1fs' % (t1-t0))

    # 1. Sample
    index = lib.sample(size=1024)
    X_train = lib.data[index]
    t2 = time.time()
    print('Sampling time: %.1fs' % (t2-t1))

    # 2. Evaluate (Interact)
    # y_train = np.zeros((X_train.shape[0]))
    y_train = evaluate(av_model, env, X_train)    # TODO: use actual y_train by evaluation
    t3 = time.time()
    print('Evaluation time: %.1fs' % (t3-t2))

    # 3. Train reward predictor
    predictor = train_predictor(predictor, X_train, y_train)
    t4 = time.time()
    print('Training reward predictor time: %.1fs' % (t4-t3))

    # 4. Labeling
    lib.labeling(predictor)
    t5 = time.time()
    print('Labeling time: %.1fs' % (t5-t4))

    # 5. Select
    index = lib.select(size=100)
    selected_scenario = lib.data[index]
    t6 = time.time()
    print('Selecting time: %.1fs' % (t6-t5))

    # 6. Train AV model
    av_model = train_av(av_model, env, selected_scenario)
    t7 = time.time()
    print('Training AV model time: %.1fs' % (t7-t6))


if __name__ == '__main__':
    main()
