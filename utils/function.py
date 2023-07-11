import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.av_policy_sac import SAC
from utils.av_policy import RL_brain
from utils.environment import Env
from utils.predictor import predictor


def evaluate(av_model: RL_brain, env: Env, scenarios: np.ndarray) -> np.ndarray:
    """Return the performance of the AV model in the given scenarios (accident: 1, otherwise: 0). """
    scenario_num = scenarios.shape[0]
    labels = np.zeros(scenario_num)
    
    with torch.no_grad():
        for i in range(scenario_num):
            state = env.reset(scenarios[i])
            done = False
            step = 0
            
            while not done:
                step += 1
                action = av_model.choose_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action, timestep=step)
                state = next_state
            
            if info == 'fail':
                labels[i] = 1
            elif info == 'succeed':
                labels[i] = 0
    
    return labels


def train_predictor(model: predictor, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=20, lr=1e-4, batch_size=64, wandb_logger=None, device='cuda') -> predictor:
    """
    Training process of supervised learning. 
    
    No validation process, no calculation of hard labels. 
    """
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_size = y_train.size
    batch_num = math.ceil(total_size/batch_size)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        
        for iteration in range(batch_num):
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            
            out = model(X)
            loss = loss_function(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss /= batch_num
        print('    Epoch:', epoch+1, ' train loss: %.4f' % total_loss)
        
        if wandb_logger is not None:
            wandb_logger.log({'Predictor loss': total_loss})
    
    return model


def train_av_online(av_model: SAC, env: Env, scenarios: np.ndarray, 
                    episodes=100, wandb_logger=None) -> SAC:
    """Training process of online reinforcement learning. """
    total_step = 0
    for episode in range(episodes):
        np.random.shuffle(scenarios)
        
        scenario_num = scenarios.shape[0]
        success_count = 0
        
        for i in range(scenario_num):
            state = env.reset(scenarios[i])
            scenario_reward = 0     # reward of each scenario
            done = False
            step = 0
            
            while not done:
                step += 1
                total_step += 1
                action = av_model.choose_action(state)
                next_state, reward, done, info = env.step(action, timestep=step)
                not_done = 0.0 if done else 1.0
                av_model.replay_buffer.store_transition((state, action, reward, next_state, not_done))
                state = next_state
                scenario_reward += reward

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
                success_count += 1
            # print('        Episode:', episode+1, ' Scenario:', i, ' Reward: %.2f ' % scenario_reward, info)
            if wandb_logger is not None:
                wandb_logger.log({
                    'episode_reward': scenario_reward, 
                    })
        
        success_rate = success_count / scenario_num
        print('    Episode:', episode+1, ' Training success rate: %.2f' % success_rate)
        if wandb_logger is not None:
            wandb_logger.log({
                'success_count': success_count, 
                'success_rate': success_rate, 
                })
    
    return av_model


def train_av(av_model: RL_brain, env: Env, scenarios: np.ndarray, 
             epochs=20, episodes=20, wandb_logger=None) -> RL_brain:
    """Training process of offline reinforcement learning. """
    for episode in range(episodes):
        # rollout & evaluate
        np.random.shuffle(scenarios)
        av_model.replay_buffer.clear()  # clear previous transitions
        
        scenario_num = scenarios.shape[0]
        success_count = 0
        
        for i in range(scenario_num):
            state = env.reset(scenarios[i])
            scenario_reward = 0     # reward of each scenario
            done = False
            step = 0
            
            while not done:
                step += 1
                action = av_model.choose_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action, timestep=step)
                not_done = 0.0 if done else 1.0
                av_model.replay_buffer.store_transition((state, action, reward, next_state, not_done))
                state = next_state
                scenario_reward += reward
            
            if info == 'succeed':
                success_count += 1
            # print('        Episode:', episode+1, ' Scenario:', i, ' Reward: %.2f ' % scenario_reward, info)
            if wandb_logger is not None:
                wandb_logger.log({
                    'episode_reward': scenario_reward, 
                    })
        
        success_rate = success_count / scenario_num
        print('    Episode:', episode+1, ' Training success rate: %.2f' % success_rate)
        if wandb_logger is not None:
            wandb_logger.log({
                'success_count': success_count, 
                'success_rate': success_rate, 
                })
        
        # train
        for epoch in range(epochs):
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
                    })
    
    return av_model
