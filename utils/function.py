import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.av_policy import SAC
from utils.environment import Env
from utils.predictor import predictor


def evaluate(av_model: SAC, env: Env, scenarios: np.ndarray) -> np.ndarray:
    """Return the performance of the AV model in the given scenarios (collision: 1, otherwise: 0). """
    size = scenarios.shape[0]
    labels = np.zeros(size)
    for i in range(size):
        state = env.reset(scenarios[i])
        done = 0
        step = 0
        
        while not done:
            step += 1
            action = av_model.choose_action(state).cpu().numpy()
            next_state, reward, done, info = env.step(action, timestep=step)
            state = next_state
        
        if info == 'fail':
            labels[i] = 1
        elif info == 'succeed':
            labels[i] = 0
    
    return labels


def train_predictor(model: predictor, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=500, lr=1e-3, batch_size=64, wandb_logger=None) -> predictor:
    """Training process of supervised learning. """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    total_size = y_train.size

    for epoch in range(epochs):
        total_loss = 0.0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        
        for iteration in range(math.ceil(total_size/batch_size)):
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
            out = model(torch.FloatTensor(X))
            y = torch.tensor(y)
            loss = loss_function(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if wandb_logger is not None:
            wandb_logger.log({'Predictor loss': total_loss/total_size})
    
    return model


def train_av(av_model: SAC, env: Env, scenarios: np.ndarray, episodes=100, wandb_logger=None) -> SAC:
    """Training process of reinforcement learning. """
    for i in range(scenarios.shape[0]):
        for episode in range(episodes):
            state = env.reset(scenarios[i])
            episode_reward = 0    # reward of each episode
            done = 0
            step = 0
            
            while not done:
                step += 1
                action = av_model.choose_action(state).cpu().numpy()
                next_state, reward, done, info = env.step(action, timestep=step)
                not_done = 0.0 if done else 1.0
                av_model.memory.store_transition((state, action, reward, next_state, not_done))
                state = next_state
                episode_reward += reward

                if episode > 10:
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
                
                if done:
                    break
            
            if wandb_logger is not None:
                wandb_logger.log({'episode_reward': episode_reward})
            print('    Episode: ', episode+1, 'Reward: %.2f' % episode_reward, info)
    
    return av_model
