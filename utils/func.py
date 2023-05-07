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
                    epochs=500, lr=1e-3) -> reward_predictor:
    """Training process of supervised learning. """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        out = model(torch.FloatTensor(X_train))
        y = torch.tensor(y_train)
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def train_av(av_model: SAC, env: Env, scenarios: np.ndarray, episodes=100) -> SAC:
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
                    av_model.train()
                
                if done:
                    break
    
    return av_model
