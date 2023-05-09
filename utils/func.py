import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.av_policy import SAC
from utils.env import Env
from utils.reward_predictor import reward_predictor


def evaluate(av_model: SAC, env: Env, scenarios: np.ndarray, size: int) -> np.ndarray:
    """Return the performance of the AV model in the given scenarios (collision: 1, otherwise: 0). """
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


def train_predictor(model: reward_predictor, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=500, lr=1e-3, wandb_logger=None) -> reward_predictor:
    """Training process of supervised learning. """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs): # TODO: train with mini-batch?
        out = model(torch.FloatTensor(X_train))
        y = torch.tensor(y_train)
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if wandb_logger is not None:
            wandb_logger.log({'Predictor loss': loss})
    
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
                        })
                
                if done:
                    break
    
    return av_model
