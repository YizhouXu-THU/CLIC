import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class PrioritizedReplayBuffer:
    """
    Define the prioritized replay buffer. \n
    The transitions with higher TD errors have higher priorities and are more likely to be sampled.
    """

    def __init__(self, capacity: int, alpha=0.6, beta=0.4, device='cuda') -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = collections.deque(maxlen=capacity)
        self.device = device
        self.priorities = np.zeros((capacity,), dtype=np.float16)
        self.priorities_sum = 0.0
        self.priorities_max = 1.0

    def size(self) -> int:
        return len(self.memory)

    def store_transition(self, data: tuple[np.ndarray, np.ndarray, float, np.ndarray, float]) -> None:
        """data: (state, action, reward, next_state, not_done)"""
        max_priority = self.priorities.max() if self.memory else self.priorities_max
        self.memory.append(data)
        self.priorities[self.size()-1] = max_priority
        self.priorities_sum += max_priority

    def sample(self, batch_size=128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                                              np.ndarray, np.ndarray]:
        """sample a batch of experience to train"""
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        not_done_list = []
        
        if self.size() == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        batch = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float16)


        for experience in batch:
            state, action, reward, next_state, not_done = experience
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            not_done_list.append(not_done)

        return torch.FloatTensor(np.array(state_list)).to(self.device), \
               torch.FloatTensor(np.array(action_list)).to(self.device), \
               torch.FloatTensor(np.array(reward_list)).unsqueeze(-1).to(self.device), \
               torch.FloatTensor(np.array(next_state_list)).to(self.device), \
               torch.FloatTensor(np.array(not_done_list)).unsqueeze(-1).to(self.device), \
               indices, torch.FloatTensor(weights).to(self.device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon=1e-6) -> None:
        """update priorities of sampled transitions"""
        priorities = np.abs(td_errors) + epsilon
        priorities = np.minimum(priorities, self.priorities_max)
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.priorities_sum = self.priorities.sum()

    def clear(self) -> None:
        self.memory.clear()
        self.priorities = np.zeros((self.capacity,), dtype=np.float16)
        self.priorities_sum = 0.0


class ScalarNet(nn.Module):
    """Scalar network for alpha optimization. """

    def __init__(self, init_value=0.0) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.constant


class ValueNet(nn.Module):
    """Critic"""

    def __init__(self, state_dim: int, edge=3e-3) -> None:
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
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

    def __init__(self, input_dim: int, edge=3e-3) -> None:
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # initialize the output layer
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)   # concatenate in transverse
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class PolicyNet(nn.Module):
    """Actor"""

    def __init__(self, state_dim: int, action_dim: int, action_range: np.ndarray, 
                 log_std_min=-20, log_std_max=2, edge=3e-3, device='cuda') -> None:
        super(PolicyNet, self).__init__()
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

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
    
    def choose_action(self, state: np.ndarray, deterministic=False) -> torch.Tensor:
        """Sample action based on state. """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        if deterministic:   # choose mean as action
            action = mean
        else:               # sampling from Gaussian distribution
            normal = Normal(mean, std)
            action = normal.sample()
        action = torch.tanh(action).detach()    # shape: [1, action_dim], range: [-1, 1]

        action_split = torch.chunk(action, chunks=2, dim=1)     # split the action into speed and yaw
        # extend the action to the actual range
        # action_abs = torch.clamp(action_split[0], self.action_range[0,0], self.action_range[0,1])
        # action_arg = torch.clamp(action_split[1], self.action_range[1,0], self.action_range[1,1])
        action_abs = (self.action_range[0,1] + self.action_range[0,0] + \
                     (self.action_range[0,1] - self.action_range[0,0]) * action_split[0]) / 2
        action_arg = (self.action_range[1,1] + self.action_range[1,0] + \
                     (self.action_range[1,1] - self.action_range[1,0]) * action_split[1]) / 2
        action = torch.cat((action_abs, action_arg), dim=1)

        return action   # shape: [1, action_dim]

    def evaluate(self, state: torch.Tensor, epsilon=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterize to calculate entropy. """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)
        z = noise.sample()  # sample noise in standard normal distribution
        z = z.to(self.device)
        
        action = torch.tanh(mean + std*z)   # shape: [batch_size, action_dim], range: [-1, 1]
        # calculate the entropy of the action
        log_prob = normal.log_prob(mean + std*z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1).unsqueeze(-1) # dimension elevate after summation
        
        action_split = torch.chunk(action, chunks=2, dim=1)     # split the action into speed and yaw
        # extend the action to the actual range
        # action_abs = torch.clamp(action_split[0], self.action_range[0,0], self.action_range[0,1])
        # action_arg = torch.clamp(action_split[1], self.action_range[1,0], self.action_range[1,1])
        action_abs = (self.action_range[0,1] + self.action_range[0,0] + \
                     (self.action_range[0,1] - self.action_range[0,0]) * action_split[0]) / 2
        action_arg = (self.action_range[1,1] + self.action_range[1,0] + \
                     (self.action_range[1,1] - self.action_range[1,0]) * action_split[1]) / 2
        action = torch.cat((action_abs, action_arg), dim=1)

        return action, log_prob # shape: [batch_size, action_dim], [batch_size, 1]


class RL_brain:
    def __init__(self, env, capacity=int(1e6), device='cuda', 
                 batch_size=128, lr=1e-4, gamma=0.99, tau=0.01, target_entropy=0.0, 
                 alpha_sac=0.2, alpha_multiplier=1.0, alpha_prb=0.6, beta=0.4, beta_multiplier=0.0, epsilon=1e-6) -> None:
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.alpha_prb = alpha_prb
        self.beta = beta
        self.beta_multiplier = beta_multiplier
        self.epsilon = epsilon

        # initialize networks
        self.value_net = ValueNet(state_dim=env.state_dim).to(device)
        self.target_value_net = ValueNet(state_dim=env.state_dim).to(device)
        self.q1_net = SoftQNet(input_dim=env.state_dim+env.action_dim).to(device)
        self.q2_net = SoftQNet(input_dim=env.state_dim+env.action_dim).to(device)
        self.policy_net = PolicyNet(state_dim=env.state_dim, action_dim=env.action_dim, 
                                    action_range=env.action_range, device=device).to(device)
        self.log_alpha = ScalarNet(init_value=np.log(alpha_sac)).to(device)

        # initialize target network (with the same form as the soft update process)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1-self.tau) * target_param)

        # initialize optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam(self.log_alpha.parameters(), lr=lr)

        # initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha_prb, beta=beta, device=device)

    def choose_action(self, state: torch.Tensor, deterministic=False) -> np.ndarray:
        action = self.policy_net.choose_action(state, deterministic=deterministic)  # shape: [1, action_dim]
        return action.squeeze().cpu().numpy() # shape: [action_dim]

    def train(self, auto_alpha=True) -> dict[str, float]:
        state, action, reward, next_state, not_done, indices, weights = self.replay_buffer.sample(self.batch_size)
        new_action, log_prob = self.policy_net.evaluate(state)

        # alpha loss function
        if auto_alpha:
            alpha_loss = -(self.log_alpha() * (log_prob + self.target_entropy).detach()).mean()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        # V value loss function
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        new_q_value = torch.min(new_q1_value, new_q2_value)
        next_value = new_q_value - alpha * log_prob
        value_loss = F.mse_loss(value, next_value.detach(), reduction='none')
        value_loss = (value_loss * weights).mean()

        # soft Q loss function
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + not_done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach(), reduction='none')
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach(), reduction='none')
        q1_value_loss = (q1_value_loss * weights).mean()
        q2_value_loss = (q2_value_loss * weights).mean()

        # policy loss function
        policy_loss = (alpha * log_prob - new_q_value).mean()
        policy_loss = torch.as_tensor(policy_loss, device=self.device)

        # update parameters
        self.value_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        if auto_alpha:
            self.alpha_optimizer.zero_grad()

        value_loss.backward()
        q1_value_loss.backward()
        q2_value_loss.backward()
        policy_loss.backward()
        if auto_alpha:
            alpha_loss.backward()

        self.value_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.policy_optimizer.step()
        if auto_alpha:
            self.alpha_optimizer.step()

        # update priorities
        td_errors = abs(new_q_value - alpha * log_prob - (q1_value.detach() + q2_value.detach()) / 2) + self.epsilon
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy(), epsilon=self.epsilon)

        # soft update of target network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1-self.tau) * target_param)

        # increase beta
        self.beta = min(self.beta + self.beta_multiplier * (1 - self.beta), 1.0)
        self.beta_multiplier += 1e-6
        self.beta_multiplier = min(self.beta_multiplier, 1.0)

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
            policy_loss=policy_loss.item(),
            alpha=alpha.item(),
            alpha_loss=alpha_loss.item(),
            beta=self.beta,
            beta_multiplier=self.beta_multiplier,
        )
