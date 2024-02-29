import math
import random
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy import interpolate

from utils.predictor import predictor_mlp
# from utils.cql.SimpleSAC.replay_buffer import batch_to_torch, subsample_batch
# from utils.cql.SimpleSAC.utils import prefix_metrics


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def focal_loss(out: torch.Tensor, y: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
    """Focal loss for binary classification. """
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')(out, y)
    pt = torch.exp(-bce_loss)
    loss = alpha * (1 - pt) ** gamma * bce_loss
    return loss.mean()


def make_dataset(av_model, env, scenarios: np.ndarray, device='cuda') -> dict[str, np.ndarray]:
    """Rollout scenarios and make a dataset for cql training."""
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    
    with torch.no_grad():
        for i in range(scenarios.shape[0]):
            state = env.reset(scenarios[i])
            done = False
            step = 0
            
            while not done:
                step += 1
                state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=device)
                action = av_model.choose_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action, timestep=step, need_reward=False)
                
                states.append(state.detach().cpu().numpy()[0])
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                
                state = next_state
    
    states = np.array(states, dtype=np.float16)
    actions = np.array(actions, dtype=np.float16)
    next_states = np.array(next_states, dtype=np.float16)
    rewards = np.array(rewards, dtype=np.float16)
    dones = np.array(dones, dtype=np.float16)
    
    dataset = dict(
        observations=states, 
        actions=actions, 
        next_observations=next_states, 
        rewards=rewards, 
        dones=dones, 
    )
    return dataset


def evaluate(av_model, env, scenarios: np.ndarray, need_metrics=False) -> tuple[dict[str, float], np.ndarray]:
    """
    Return the metrics of the AV model in the given scenarios. 
    
    metrics: 
        sr: Success rate. 
        cps: Average collision frequency per second. 
        cpm: Average collision frequency per meter. 
        
        vel: The arithmetic mean of the absolute value of AV velocity at each timestep. 
        acc: The arithmetic mean of the absolute value of AV acceleration at each timestep. 
        jerk: The arithmetic mean of the absolute value of AV jerk at each timestep. 
        ang_vel: The arithmetic mean of the absolute value of AV angular velocity at each timestep. 
        lateral_acc: The arithmetic mean of the absolute value of AV lateral acceleration at each timestep. 
        
        success_vel: The arithmetic mean of the absolute value of AV velocity at each timestep in successful scenarios. 
    
    labels: The performance of the AV model in the given scenarios (accident: 1, otherwise: 0). 
    """
    scenario_num = scenarios.shape[0]
    labels = np.zeros(scenario_num, dtype=int)
    if need_metrics:
        vels, accs, jerks, ang_vels, lateral_accs, success_vel = \
                        np.empty((0)), np.empty((0)), np.empty((0)), np.empty((0)), np.empty((0)), np.empty((0))
        total_time, total_dist = 0.0, 0.0
    # print('    evaluating...')
    
    with torch.no_grad():
        for i in range(scenario_num):
        # for i in trange(scenario_num):
            vel, yaw = [], []
            state = env.reset(scenarios[i])
            done = False
            step = 0
            x_init = state[0]
            
            while not done:
                vel.append(state[2])
                yaw.append(state[3])
                step += 1
                if av_model is not None:
                    action = av_model.choose_action(state, deterministic=True)
                else:   # random policy
                    action = np.array([np.random.uniform(range[0], range[1]) for range in env.action_range])
                next_state, reward, done, info = env.step(action, timestep=step, need_reward=False)
                state = next_state
            
            if info == 'fail':
                labels[i] = 1
            elif info == 'succeed':
                labels[i] = 0
            
            if need_metrics:
                total_time += (step * env.delta_t)
                total_dist += (state[0] - x_init)
                vel, yaw = np.array(vel), np.array(yaw)
                acc = np.diff(vel) / env.delta_t
                jerk = np.diff(acc) / env.delta_t
                ang_vel = np.diff(yaw) / env.delta_t
                lateral_acc = vel[1:] * ang_vel
                
                vels = np.concatenate((vels, np.abs(vel)))
                accs = np.concatenate((accs, np.abs(acc)))
                jerks = np.concatenate((jerks, np.abs(jerk)))
                ang_vels = np.concatenate((ang_vels, np.abs(ang_vel)))
                lateral_accs = np.concatenate((lateral_accs, np.abs(lateral_acc)))
                if info == 'succeed':
                    success_vel = np.concatenate((success_vel, np.abs(vel)))
    
    if need_metrics:
        col_num = np.sum(labels)
        
        sr = 1 - col_num / labels.size
        cps = col_num / total_time
        cpm = col_num / total_dist
        
        vel = np.mean(vels)
        acc = np.mean(accs)
        jerk = np.mean(jerks)
        ang_vel = np.mean(ang_vels)
        lateral_acc = np.mean(lateral_accs)
        success_vel = np.mean(success_vel)
        
        metrics = dict(
            sr=sr, 
            cps=cps, 
            cpm=cpm, 
            vel=vel, 
            acc=acc, 
            jerk=jerk, 
            ang_vel=ang_vel, 
            lateral_acc=lateral_acc, 
            success_vel=success_vel, 
        )
    else:
        metrics = {}
    
    return metrics, labels


def train_predictor(predictor, 
                    X_train: np.ndarray, y_train: np.ndarray, 
                    epochs=20, lr=1e-4, batch_size=128, 
                    wandb_logger=None, device='cuda') -> None:
    """
    Training process of supervised learning. \n
    No validation process, no calculation of hard labels. 
    """
    loss_function = nn.BCELoss()
    # loss_function = focal_loss
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_size = y_train.size

    for epoch in range(epochs):
        predictor.train()
        batch_num = math.ceil(total_size/batch_size)
        total_loss = 0.0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            
            out = predictor(X)
            loss = loss_function(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss /= batch_num
        
        print('    ', 'Epoch:', epoch+1, ' train loss: %.4f' % total_loss)
        if wandb_logger is not None:
            wandb_logger.log({
                'Predictor loss': total_loss
                })


def train_validate_predictor(predictor, 
                             X_train: np.ndarray, y_train: np.ndarray, 
                             X_valid: np.ndarray, y_valid: np.ndarray, 
                             epochs=20, lr=1e-4, batch_size=128, 
                             wandb_logger=None, device='cuda') -> None:
    """
    Training process of supervised learning. \n
    Including validation process, but still no calculation of hard labels. 
    """
    loss_function = nn.BCELoss()
    # loss_function = focal_loss
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    # optimizer = optim.SGD(predictor.parameters(), lr=lr, momentum=0.9)
    train_size = y_train.size
    valid_size = y_valid.size

    for epoch in range(epochs):
        # train
        predictor.train()
        batch_num = math.ceil(train_size/batch_size)
        total_loss = 0.0
        total_correct = 0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), -1]
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            
            out = predictor(X)
            loss = loss_function(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred = (out > 0.5).data.cpu().numpy().squeeze()
            y = y.data.cpu().numpy().squeeze()
            total_correct += sum(y_pred == y)
        
        total_loss /= batch_num
        accuracy = total_correct / train_size
        
        print('\nEpoch:', epoch+1, ' train loss: %.4f' % total_loss, ' train accuracy: %.4f' % accuracy, end='    ')
        if wandb_logger is not None:
            wandb_logger.log({
                'Predictor train loss': total_loss, 
                'Predictor train accuracy': accuracy, 
                })
        
        # validate
        with torch.no_grad():
            predictor.eval()
            batch_num = math.ceil(valid_size/batch_size)
            out = torch.zeros(0, dtype=torch.float32, device=device)
            X = torch.tensor(X_valid, dtype=torch.float32, device=device)
            y = torch.tensor(y_valid, dtype=torch.float32, device=device)
            # out = predictor(X)
            for iteration in range(batch_num):
                X_batch = X_valid[iteration*batch_size : min((iteration+1)*batch_size,valid_size)]
                X_batch = torch.tensor(X_batch, dtype=torch.float32, device=device)
                out_batch = predictor(X_batch)
                out = torch.cat((out, out_batch), dim=0)
            bce_loss = loss_function(out, y)
        
        y_pred = (out > 0.5).data.cpu().numpy().squeeze()
        accuracy = sum(y_pred == y_valid) / valid_size
        
        print(' test loss: %.4f' % bce_loss.item(), ' test accuracy: %.4f' % accuracy, end='')
        if wandb_logger is not None:
            wandb_logger.log({
                'Predictor valid loss': loss.item(), 
                'Predictor valid accuracy': accuracy, 
                })


def train_predictor_vae(vae, classifier, scenario_lib, 
                        X_train: np.ndarray, y_train: np.ndarray, 
                        epochs_vae=1000, epochs=20, lr=1e-4, batch_size=128, 
                        wandb_logger=None, device='cuda') -> None:
    # train VAE for reconstruction
    def loss_vae(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    total_size = y_train.size
    
    for epoch in range(epochs_vae):
        vae.train()
        batch_num = math.ceil(total_size/batch_size)
        total_loss = 0.0
        total_mse_loss = 0.0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
            # y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
            X = scenario_lib.scenario_normalize(X)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            # y = torch.tensor(y, dtype=torch.float32, device=device)
            
            decoded, mu, log_var = vae(X)
            loss = loss_vae(decoded, X, mu, log_var)
            total_loss += loss.item()
            total_mse_loss += F.mse_loss(decoded, X).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss /= batch_num
        total_mse_loss /= batch_num
        
        print('Epoch:', epoch+1, ' VAE loss: %.1f' % total_loss, ' VAE MSE loss: %.6f' % total_mse_loss)
        if wandb_logger is not None:
            wandb_logger.log({
                'VAE loss': total_loss, 
                'VAE MSE loss': total_mse_loss, 
                })
    
    
    # train predictor for classification
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    loss_classifier = nn.BCELoss()
    
    vae.eval()
    for epoch in range(epochs):
        classifier.train()
        batch_num = math.ceil(total_size/batch_size)
        total_loss = 0.0
        total_correct = 0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,total_size), -1]
            X = scenario_lib.scenario_normalize(X)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            
            _, latent, _ = vae(X)
            out = classifier(latent)
            loss = loss_classifier(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred = (out > 0.5).data.cpu().numpy().squeeze()
            y = y.data.cpu().numpy().squeeze()
            total_correct += sum(y_pred == y)
        
        total_loss /= batch_num
        accuracy = total_correct / total_size
        
        print('Epoch:', epoch+1, ' classifier loss: %.4f' % total_loss, ' classifier accuracy: %.4f' % accuracy)
        if wandb_logger is not None:
            wandb_logger.log({
                'Classifier loss': total_loss, 
                'Classifier accuracy': accuracy, 
                })


def train_validate_predictor_vae(vae, classifier, scenario_lib, 
                                 X_train: np.ndarray, y_train: np.ndarray, 
                                 X_valid: np.ndarray, y_valid: np.ndarray,
                                 epochs_vae=1000, epochs=20, lr=1e-4, batch_size=128, 
                                 wandb_logger=None, device='cuda') -> None:
    # train and validate VAE for reconstruction
    def loss_vae(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    train_size = y_train.size
    valid_size = y_valid.size
    
    for epoch in range(epochs_vae):
        # train
        vae.train()
        batch_num = math.ceil(train_size/batch_size)
        total_loss = 0.0
        total_mse_loss = 0.0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), 0:-1]
            # y = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), -1]
            X = scenario_lib.scenario_normalize(X)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            # y = torch.tensor(y, dtype=torch.float32, device=device)
            
            decoded, mu, log_var = vae(X)
            loss = loss_vae(decoded, X, mu, log_var)
            total_loss += loss.item()
            total_mse_loss += F.mse_loss(decoded, X).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss /= batch_num
        total_mse_loss /= batch_num
        
        print('\nEpoch:', epoch+1, ' VAE train loss: %.4f' % total_loss, ' VAE train MSE loss: %.4f' % total_mse_loss, end='    ')
        if wandb_logger is not None:
            wandb_logger.log({
                'VAE train loss': total_loss, 
                'VAE train MSE loss': total_mse_loss, 
                })
        
        # validate
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                vae.eval()
                batch_num = math.ceil(valid_size/batch_size)
                decoded = torch.zeros((0, X_valid.shape[1]), dtype=torch.float32, device=device)
                mu = torch.zeros((0, vae.latent_dim), dtype=torch.float32, device=device)
                log_var = torch.zeros((0, vae.latent_dim), dtype=torch.float32, device=device)
                X_valid = scenario_lib.scenario_normalize(X_valid)
                X = torch.tensor(X_valid, dtype=torch.float32, device=device)
                y = torch.tensor(y_valid, dtype=torch.float32, device=device)
                # decoded, output, mu, log_var = predictor(X)
                for iteration in range(batch_num):
                    X_batch = X_valid[iteration*batch_size : min((iteration+1)*batch_size,valid_size)]
                    X_batch = torch.tensor(X_batch, dtype=torch.float32, device=device)
                    decoded_batch, mu_batch, log_var_batch = vae(X_batch)
                    decoded = torch.cat((decoded, decoded_batch), dim=0)
                    mu = torch.cat((mu, mu_batch), dim=0)
                    log_var = torch.cat((log_var, log_var_batch), dim=0)
                loss = loss_vae(decoded, X, mu, log_var)
                mse_loss = F.mse_loss(decoded, X)
            
            print(' VAE valid loss: %.4f' % loss.item(), ' VAE valid MSE loss: %.4f' % mse_loss.item(), end='')
            if wandb_logger is not None:
                wandb_logger.log({
                    'VAE valid loss': loss.item(), 
                    'VAE valid MSE loss': mse_loss.item(),  
                    })
    print()
    
    
    # train and validate predictor for classification
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    loss_classifier = nn.BCELoss()
    
    vae.eval()
    for epoch in range(epochs):
        # train
        classifier.train()
        batch_num = math.ceil(train_size/batch_size)
        total_loss = 0.0
        total_correct = 0
        # shuffle
        data_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        np.random.shuffle(data_train)
        # data_pos, data_neg = data_train[data_train[:,-1]==1], data_train[data_train[:,-1]==0]
        
        for iteration in range(batch_num):
            # batch_pos = data_pos[np.random.choice(data_pos.shape[0], batch_size//2, replace=False)]
            # batch_neg = data_neg[np.random.choice(data_neg.shape[0], batch_size//2, replace=False)]
            # batch = np.concatenate((batch_pos, batch_neg), axis=0)
            # np.random.shuffle(batch)
            # X, y = batch[:, 0:-1], batch[:, -1]
            X = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), 0:-1]
            y = data_train[iteration*batch_size : min((iteration+1)*batch_size,train_size), -1]
            X = scenario_lib.scenario_normalize(X)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            
            decoded, mu, log_var = vae(X)
            out = classifier(mu)
            loss = loss_classifier(out, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred = (out > 0.5).data.cpu().numpy().squeeze()
            y = y.data.cpu().numpy().squeeze()
            total_correct += sum(y_pred == y)
        
        total_loss /= batch_num
        accuracy = total_correct / train_size
        
        print('Epoch:', epoch+1, ' classifier train loss: %.4f' % total_loss, ' classifier train accuracy: %.4f' % accuracy, end='    ')
        if wandb_logger is not None:
            wandb_logger.log({
                'Classifier train loss': total_loss, 
                'Classifier train accuracy': accuracy, 
                })
        
        # validate
        with torch.no_grad():
            classifier.eval()
            batch_num = math.ceil(valid_size/batch_size)
            out = torch.zeros(0, dtype=torch.float32, device=device)
            X_valid = scenario_lib.scenario_normalize(X_valid)
            X = torch.tensor(X_valid, dtype=torch.float32, device=device)
            y = torch.tensor(y_valid, dtype=torch.float32, device=device)
            # out = predictor(X)
            for iteration in range(batch_num):
                X_batch = X_valid[iteration*batch_size : min((iteration+1)*batch_size,valid_size)]
                X_batch = torch.tensor(X_batch, dtype=torch.float32, device=device)
                _, latent, _ = vae(X_batch)
                out_batch = classifier(latent)
                out = torch.cat((out, out_batch), dim=0)
            bce_loss = loss_classifier(out, y)
        
        y_pred = (out > 0.5).data.cpu().numpy().squeeze()
        accuracy = sum(y_pred == y_valid) / valid_size
        
        print(' classifier valid loss: %.4f' % bce_loss.item(), ' classifier valid accuracy: %.4f' % accuracy)
        if wandb_logger is not None:
            wandb_logger.log({
                'Classifier valid loss': bce_loss.item(), 
                'Classifier valid accuracy': accuracy, 
                })


def train_av_online(av_model, env, scenarios: np.ndarray, 
                    episodes=100, auto_alpha=True, wandb_logger=None) -> None:
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
                action = av_model.choose_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action, timestep=step)
                not_done = 0.0 if done else 1.0
                av_model.replay_buffer.store_transition((state, action, reward, next_state, not_done))
                state = next_state
                scenario_reward += reward

                if total_step > 2 * av_model.batch_size:
                    logger = av_model.train(auto_alpha)
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
            # print('        Episode:', episode+1, ' Scenario:', i, ' Reward: %.3f ' % scenario_reward, info)
            if wandb_logger is not None:
                wandb_logger.log({
                    'scenario_reward': scenario_reward, 
                    })
        
        success_rate = success_count / scenario_num
        print('    Episode:', episode+1, ' Training success rate: %.4f' % success_rate)
        if wandb_logger is not None:
            wandb_logger.log({
                'success_count': success_count, 
                'success_rate': success_rate, 
                })


def train_av_offline(av_model, env, scenarios: np.ndarray, 
                     epochs=20, episodes=20, auto_alpha=True, wandb_logger=None) -> None:
    """Training process of offline reinforcement learning. """
    for episode in range(episodes):
        # rollout & evaluate
        np.random.shuffle(scenarios)
        av_model.replay_buffer.clear()  # clear previous transitions
        
        scenario_num = scenarios.shape[0]
        success_count = 0
        total_reward = 0
        
        for i in range(scenario_num):
            state = env.reset(scenarios[i])
            scenario_reward = 0     # reward of each scenario
            done = False
            step = 0
            
            while not done:
                step += 1
                action = av_model.choose_action(state, deterministic=True)
                next_state, reward, done, info = env.step(action, timestep=step)
                not_done = 0.0 if done else 1.0
                av_model.replay_buffer.store_transition((state, action, reward, next_state, not_done))
                state = next_state
                scenario_reward += reward
            
            if info == 'succeed':
                success_count += 1
            # print('        Episode:', episode+1, ' Scenario:', i, ' Reward: %.3f ' % scenario_reward, info)
            if wandb_logger is not None:
                wandb_logger.log({
                    'scenario_reward': scenario_reward, 
                    })
            total_reward += scenario_reward
        
        success_rate = success_count / scenario_num
        average_reward = total_reward / scenario_num
        print('    Episode:', episode+1, ' Training success rate: %.4f' % success_rate, ' Average reward: %.3f' % average_reward)
        if wandb_logger is not None:
            wandb_logger.log({
                'success_count': success_count, 
                'success_rate': success_rate, 
                'average_reward': average_reward, 
                })
        
        # train
        for epoch in range(epochs):
            logger = av_model.train(auto_alpha)
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


# def train_cql(av_model, env, scenarios: np.ndarray, FLAGS, wandb_logger) -> None:
#     """Training process of CQL reinforcement learning algorithm. """
#     dataset = make_dataset(av_model, env, scenarios, device=FLAGS.device)
#     for epoch in range(FLAGS.n_epochs):
#         metrics = {'epoch': epoch}

#         for batch_idx in range(FLAGS.n_train_step_per_epoch):
#             batch = subsample_batch(dataset, FLAGS.batch_size)
#             batch = batch_to_torch(batch, FLAGS.device)
#             metrics.update(prefix_metrics(av_model.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

#         wandb_logger.log(metrics)


def cm_result(label_before: np.ndarray, label_after: np.ndarray) -> None:
    """
    Calculate the confusion matrix of the label before and after training, 
    which gives the number and proportion of scenarios 
    where success or failure has changed or remained unchanged after training. 
    
    Each row of the confusion matrix represents the state before training, 
    and each column represents the state after training. 
    
    Label 0 indicates success, and label 1 indicates failure. 
    """
    print('Confusion matrix (quantity):')
    cm = confusion_matrix(label_before, label_after)
    print(cm)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # convert to percentage
    np.set_printoptions(precision=3)
    print('Confusion matrix (proportion):')
    print(cm)


def matrix_test(predictor_params: list, policy_net_params: list, 
                av_model, predictor, env, scenario_lib, test_size=4096, device='cuda') -> np.ndarray:
    """
    Conduct matrix testing on the combination of predictor models and AV models for each stage, 
    label each scenario with the predictor, select key scenarios for testing the AV model, 
    and provide the success rate of the testing. 

    Return a numpy array with the shape of ((rounds+1), rounds). 
    Each row represents an AV model, with the total number of (rounds+1) (including untrained original models); 
    each column represents a predictor model, with the total number of rounds. 
    """
    results = np.zeros((len(policy_net_params), len(predictor_params)))
    # matrix test and get results
    for i, policy_net_param in enumerate(policy_net_params):
        av_model.policy_net.load_state_dict(policy_net_param)
        
        for j, predictor_param in enumerate(predictor_params):
            predictor.load_state_dict(predictor_param)
            
            scenario_lib.labeling(predictor)
            index = scenario_lib.select(size=test_size)
            test_scenario = scenario_lib.data[index]
            
            _, label = evaluate(av_model, env, test_scenario)
            results[i, j] = 1 - np.sum(label) / test_size
    
    return results


def draw_surface(data: np.ndarray, precision=0.05, smooth=False, save_path='./figure/', filename='3D_matrix.png') -> None:
    """Draw a 3D surface graph of the 2D result matrix, with smoothing operation optional. """
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x = np.arange(0, data.shape[1], 1)
    y = np.arange(0, data.shape[0], 1)
    
    if smooth:
        f = interpolate.interp2d(x, y, data, kind='cubic')
        x = np.arange(0, data.shape[1]-1+precision, precision)
        y = np.arange(0, data.shape[0]-1+precision, precision)
        Z = f(x, y)
    else:
        Z = data
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z, alpha=1, cstride=1, rstride=1, cmap='rainbow')

    ax.set_xlabel('predictor round')
    ax.set_ylabel('av round')
    ax.set_zlabel('success rate')
    ax.set_title('Success Rate of matrix test')
    fig.colorbar(surf, shrink=0.6, aspect=10, location='left')

    plt.draw()
    plt.savefig(save_path + filename)
