import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.av_policy import SAC
from utils.env import Env
from utils.reward_predictor import reward_predictor


def evaluate(av_model: SAC, env: Env, X_train: np.ndarray, av_speed: np.ndarray) -> np.ndarray:
    y_train = []
    for i in range(X_train.shape[0]):
        scenario = X_train[i]
        state = env.reset(scenario, av_speed[i])
        done = 0
        step = 0

        while not done:
            step += 1
            action = av_model.choose_action(state).cpu().numpy()
            next_state, reward, done, info = env.step(action, timestep=step)
            state = next_state
        
        if info == 'crash':
            y_train.append(1)
        elif info == 'arrive':
            y_train.append(0)
    
    return np.array(y_train)


def train_predictor(model: reward_predictor, X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=500, lr=1e-3) -> reward_predictor:
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


def train_av(av_model: SAC, env: Env, scenarios: np.ndarray, av_speed: np.ndarray, episodes=100) -> SAC:
    for i in range(scenarios.shape[0]):
        scenario = scenarios[i]
        for episode in range(episodes):
            state = env.reset(scenario, av_speed[i])
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
