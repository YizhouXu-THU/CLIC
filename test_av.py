from datetime import datetime
import wandb
import torch

from utils.scenario_lib import scenario_lib
from utils.av_policy import SAC
from utils.environment import Env


eval_size = 4096
train_size = 100
episodes = 100
learning_rate = 1e-4
use_wandb = True
sumo_gui = False
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

lib = scenario_lib(path='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/scenario_lib/', 
                   npy_path='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/all_data.npy')
env = Env(max_bv_num=lib.max_bv_num, 
          cfg_sumo='/home/xuyizhou/CL-for-Autonomous-Vehicle-Training-and-Testing/config/lane.sumocfg', 
          gui=sumo_gui)
av_model = SAC(env, device=device)

if use_wandb:
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=datetime.now().strftime('%Y%m%d-%H%M')+'-test_av', # for example: '20230509-1544-test_av'
        reinit=True, 
        )
else:
    wandb_logger = None

index = lib.sample(size=train_size)
train_scenario = lib.data[index]
total_step = 0

for i in range(train_size):
    print('Scenario:', i)
    scenario_success_count = 0
    scenario_success_rate = 0.0

    for episode in range(episodes):
        state = env.reset(train_scenario[i])
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            step += 1
            total_step += 1
            action = av_model.choose_action(state)
            next_state, reward, done, info = env.step(action, timestep=step)
            not_done = 0.0 if done else 1.0
            av_model.memory.store_transition((state, action, reward, next_state, not_done))
            state = next_state
            episode_reward += reward
            
            if total_step > 200:
                logger = av_model.train()
                if wandb_logger is not None:
                        wandb_logger.log({
                            'log_prob': logger['log_prob'], 
                            'value': logger['value'], 
                            'new_q1_value': logger['new_q1_value'], 
                            'new_q2_value': logger['new_q2_value'], 
                            'next_value': logger['next_value'], 
                            'value_loss': logger['value_loss'], 
                            'q1_value': logger['q1_value'], 
                            'q2_value': logger['q2_value'], 
                            'target_value': logger['target_value'], 
                            'target_q_value': logger['target_q_value'], 
                            'q1_value_loss': logger['q1_value_loss'], 
                            'q2_value_loss': logger['q2_value_loss'], 
                            'policy_loss': logger['policy_loss'], 
                            'alpha': logger['alpha'], 
                            'alpha_loss': logger['alpha_loss'], 
                            'reward': reward, 
                        })

        if info == 'succeed':
            scenario_success_count += 1
        scenario_success_rate = scenario_success_count / (episode+1)
        print('    Episode: ', episode+1, 'Reward: %.2f' % episode_reward, info)
        if wandb_logger is not None:
            wandb_logger.log({
                'scenario_success_count': scenario_success_count, 
                'scenario_success_rate': scenario_success_rate, 
                'episode_reward': episode_reward, 
                })

env.close()
