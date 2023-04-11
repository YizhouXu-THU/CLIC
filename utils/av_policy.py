import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
BATCH_SIZE = 128
LR_Q = 3e-4
LR_VALUE = 3e-4
LR_POLICY = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.01
TARGET_ENTROPY = 0.0
ALPHA_MULTIPLIER = 1.0
MEMORY_CAPACITY = 100000


class Memory:
    """
    Define the replay buffer. 
    
    First in first out, which means automatically replace the oldest memory when the memory buffer is full. 
    """

    def __init__(self, capacity=MEMORY_CAPACITY) -> None:
        self.memory = collections.deque(maxlen=capacity)

    def store_transition(self, data: tuple[np.ndarray, np.ndarray, float, np.ndarray, float]) -> None:
        self.memory.append(data)
    
    def memory_len(self) -> int:
        return len(self.memory)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample a batch of experience to train"""

        # initialize temporary container
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        not_done_list = []

        batch_size = min(BATCH_SIZE, len(self.memory))
        batch = random.sample(self.memory, batch_size)
        
        # put the experience into the corresponding container according to type
        for experience in batch:
            state, action, reward, next_state, not_done = experience
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            not_done_list.append(not_done)

        return torch.FloatTensor(np.array(state_list)).to(device), \
               torch.FloatTensor(np.array(action_list)).to(device), \
               torch.FloatTensor(np.array(reward_list)).unsqueeze(-1).to(device), \
               torch.FloatTensor(np.array(next_state_list)).to(device), \
               torch.FloatTensor(np.array(not_done_list)).unsqueeze(-1).to(device)


class ScalarNet(nn.Module):
    """Scalar network for alpha optimization. """

    def __init__(self, init_value=0.0) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.constant


# TODO: solve the problem of different values of state_dim, fill with 0?
class ValueNet(nn.Module):
    """Critic"""

    def __init__(self, env, edge=3e-3) -> None:
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(env.state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # initialize the output layer
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class SoftQNet(nn.Module):
    """Soft Q"""

    def __init__(self, env, edge=3e-3) -> None:
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(env.state_dim + env.action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # initialize the output layer
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)   # concatenate in transverse
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class PolicyNet(nn.Module):
    """Actor"""

    def __init__(self, env, log_std_min=-20, log_std_max=2, edge=3e-3) -> None:
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = env.action_range

        self.linear1 = nn.Linear(env.state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, env.action_dim)
        self.log_std_linear = nn.Linear(256, env.action_dim)

        # initialize the output layer
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)
        
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # limit the range of log_std
        
        return mean, log_std
    
    def choose_action(self, state: np.ndarray) -> torch.Tensor:
        """Sample action based on state. """

        state = torch.FloatTensor(state).to(device) # transform state to a tensor
        mean, log_std = self.forward(state)
        std = log_std.exp()    
        normal = Normal(mean, std)  # construct normal distribution for action sampling    
        action = normal.sample()    # sample action in the generated normal distribution; shape: action_dim
        
        # # use tanh() activation, with range [-1,1]; numpy can only be operated on CPU
        # action = torch.tanh(action).detach().cpu().numpy()
        # extend the action to the actual range
        # action = (self.action_range[1] + self.action_range[0] +\
        #                 (self.action_range[1] - self.action_range[0]) * action) / 2.0

        action_split = torch.chunk(action/10, chunks=2, dim=0)  # split the action into abs and arg
        action_abs = torch.clamp(action_split[0], self.action_range[0], self.action_range[1])
        action_arg = torch.clamp(action_split[1], self.action_range[2], self.action_range[3])
        action = torch.cat((action_abs, action_arg), dim=0) # shape: action_dim

        return action

    def evaluate(self, state: torch.Tensor, epsilon=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterize to calculate entropy. """

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)
        z = noise.sample()  # sample noise in standard normal distribution
        
        # action = torch.tanh(mean + std*z.to(device))  # tanh() limits the action range to [-1,1]
        # calculate the entropy of the action
        # log_prob = normal.log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = torch.sum(log_prob, dim=1).unsqueeze(-1) # dimension elevate after summation
        # extend the action to the actual range
        # action = (self.action_range[1] + self.action_range[0] +\
        #                 (self.action_range[1] - self.action_range[0]) * action) / 2.0
        
        action = mean + std*z.to(device)    # shape: batch_size * action_dim
        # calculate the entropy of the action
        log_prob = normal.log_prob(action) - torch.log(1 - torch.tanh(action).pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1).unsqueeze(-1) # dimension elevate after summation
        
        action_split = torch.chunk(action/10, chunks=2, dim=1)  # split the action into abs and arg
        action_abs = torch.clamp(action_split[0], self.action_range[0], self.action_range[1])
        action_arg = torch.clamp(action_split[1], self.action_range[2], self.action_range[3])
        action = torch.cat((action_abs, action_arg), dim=1) # shape: batch_size * action_dim

        return action, log_prob


class SAC:
    def __init__(self, env) -> None:
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

        # initialize networks
        self.value_net = ValueNet(env).to(device)
        self.target_value_net = ValueNet(env).to(device)
        self.q1_net = SoftQNet(env).to(device)
        self.q2_net = SoftQNet(env).to(device)
        self.policy_net = PolicyNet(env).to(device)
        self.log_alpha = ScalarNet(init_value=-0.001).to(device)
        
        # initialize target network (with the same form as the soft update process)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(TAU * param + (1-TAU) * target_param)

        # initialize optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LR_VALUE)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=LR_Q)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=LR_Q)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LR_POLICY)
        self.alpha_optimizer = optim.Adam(self.log_alpha.parameters(), lr=LR_ALPHA)

        # initialize replay buffer
        self.memory = Memory()

    def choose_action(self, state: np.ndarray) -> torch.Tensor:
        action = self.policy_net.choose_action(state)
        return action

    def train(self) -> dict[str, float]:
        state, action, reward, next_state, not_done = self.memory.sample()
        new_action, log_prob = self.policy_net.evaluate(state)

        # alpha loss function
        alpha_loss = -(self.log_alpha() * (log_prob + TARGET_ENTROPY).detach()).mean()
        alpha = self.log_alpha().exp() * ALPHA_MULTIPLIER

        # V value loss function
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - alpha * log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # soft Q loss function
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + not_done * GAMMA * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # policy loss function
        policy_loss = torch.Tensor((alpha * log_prob - torch.min(new_q1_value, new_q2_value)).mean())

        # update parameters
        self.value_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

        value_loss.backward()
        q1_value_loss.backward()
        q2_value_loss.backward()
        policy_loss.backward()
        alpha_loss.backward()

        self.value_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.policy_optimizer.step()
        self.alpha_optimizer.step()
        
        # soft update of target network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(TAU * param + (1-TAU) * target_param)

        # record the values during the training process
        return dict(
            log_prob=log_prob.mean().item(),
            value=value.mean().item(),
            new_q1_value=new_q1_value.mean().item(),
            new_q2_value=new_q2_value.mean().item(),
            next_value=next_value.mean().item(),
            value_loss=value_loss.item(),
            q1_value=q1_value.mean().item(),
            q2_value=q2_value.mean().item(),
            target_value=target_value.mean().item(),
            target_q_value=target_q_value.mean().item(),
            q1_value_loss=q1_value_loss.item(),
            q2_value_loss=q2_value_loss.item(),
            alpha=alpha.item(),
            alpha_loss=alpha_loss.item(),
            policy_loss=policy_loss.item(),
        )
