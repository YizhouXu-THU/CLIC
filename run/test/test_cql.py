import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import time
from copy import deepcopy
import numpy as np
import torch

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn
from utils.environment import Env
from utils.function import train_predictor, evaluate, train_cql, cm_result, matrix_test

import absl.app
import absl.flags

from utils.cql.SimpleSAC.conservative_sac import ConservativeSAC
from utils.cql.SimpleSAC.model import TanhGaussianPolicy, FullyConnectedQFunction
from utils.cql.SimpleSAC.utils import define_flags_with_default, set_random_seed, get_user_flags
from utils.cql.SimpleSAC.utils import WandBLogger


# 0. Prepare
t0 = time.time()

FLAGS_DEF = define_flags_with_default(
    max_traj_length=1000,
    # seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_model=False,
    sumo_gui = False,
    reward_type = 'r3',
    
    eval_size = 4096,
    train_size = 128,
    batch_size=256,
    rounds = 10,
    epochs = 20,
    learning_rate = 1e-4,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=10,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)
FLAGS_DEF['logging'].online = False
device = torch.device(FLAGS_DEF['device'])


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    
    # set_random_seed(FLAGS.seed)
    
    lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
    predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
    predictor.to(device)
    env = Env(max_bv_num=lib.max_bv_num, cfg_sumo='./config/lane.sumocfg', 
              gui=FLAGS.sumo_gui, reward_type=FLAGS.reward_type, seed=FLAGS.seed)
    
    policy = TanhGaussianPolicy(
        env.state_dim,
        env.action_dim,
        env.action_range,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        env.state_dim,
        env.action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        env.state_dim,
        env.action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)
    
    av_model = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2, device=device)
    av_model.torch_to_device(FLAGS.device)
    
    predictor_params = []
    policy_net_params = [deepcopy(av_model.policy_net.state_dict())]
    eval_success_rate = np.zeros(FLAGS.rounds)

    t1 = time.time()
    print('Preparation time: %.1fs' % (t1-t0))

    # test on all scenarios
    metrics_before, gt_label_before = evaluate(av_model, env, scenarios=lib.data, need_metrics=True)
    success_rate_before = 1 - np.sum(gt_label_before) / gt_label_before.size
    print('Success rate before training: %.4f\n' % success_rate_before, metrics_before)
    t2 = time.time()
    print('Evaluation time: %.1fs' % (t2-t1))

    # main loop
    for round in range(FLAGS.rounds):
        if FLAGS.logging.online:
            wandb_logger.log({'Round': round})
        print('Round %d' % (round+1))
        t2 = time.time()
        
        # 1. Sample
        index = lib.sample(size=FLAGS.eval_size)
        X_train = lib.data[index]
        t3 = time.time()
        print('    Sampling time: %.1fs' % (t3-t2))

        # 2. Evaluate (Interact)
        _, y_train = evaluate(av_model, env, scenarios=X_train)
        success_rate = 1 - np.sum(y_train) / FLAGS.eval_size
        eval_success_rate[round] = success_rate
        if FLAGS.logging.online:
            wandb_logger.log({'Evaluate success rate': success_rate})
        print('    Evaluate success rate: %.4f' % success_rate)
        t4 = time.time()
        print('    Evaluation time: %.1fs' % (t4-t3))

        # 3. Train reward predictor
        train_predictor(predictor, X_train, y_train, epochs=FLAGS.epochs, lr=FLAGS.learning_rate, 
                        batch_size=FLAGS.batch_size, wandb_logger=wandb_logger, device=device)
        predictor_params.append(deepcopy(predictor.state_dict()))
        t5 = time.time()
        print('    Training reward predictor time: %.1fs' % (t5-t4))

        # 4. Labeling
        lib.labeling(predictor)
        t6 = time.time()
        print('    Labeling time: %.1fs' % (t6-t5))

        # 5. Select
        index = lib.select(size=FLAGS.train_size)
        train_scenario = lib.data[index]
        t7 = time.time()
        print('    Selecting time: %.1fs' % (t7-t6))

        # 6. Train AV model
        train_cql(av_model, env, train_scenario, FLAGS, wandb_logger)
        policy_net_params.append(deepcopy(av_model.policy_net.state_dict()))
        t8 = time.time()
        print('    Training AV model time: %.1fs' % (t8-t7))
        
        if FLAGS.logging.online:
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
    results = matrix_test(predictor_params, policy_net_params, av_model, predictor, env, lib, FLAGS.eval_size, 
                          filename='3D_matrix_cql.png')

    np.set_printoptions(precision=4)
    print('Matrix experiment results:\n', results)

    t10 = time.time()
    print('Matrix experiment time: %.1fs' % (t10-t9))
    print('Total time: %.1fs' % (t10-t0))

    env.close()


if __name__ == '__main__':
    absl.app.run(main)
