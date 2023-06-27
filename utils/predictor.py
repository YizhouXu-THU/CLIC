import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class predictor(nn.Module):
    def __init__(self, num_input: int, num_hidden=256, num_output=2, device='cuda', dropout=False) -> None:
        super().__init__()
        # self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if isinstance(x, np.ndarray):
        #     x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = 2 * torch.tanh(self.fc3(x)) # soft clip
        output = F.softmax(x, dim=-1)[:,-1]
        return output
