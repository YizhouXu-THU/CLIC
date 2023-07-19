import torch
import torch.nn as nn
import torch.nn.functional as F


class predictor_rnn(nn.Module):
    def __init__(self, timestep: int, num_input: int, num_hidden=256, num_output=2, device='cuda') -> None:
        super().__init__()
        self.timestep = timestep
        self.num_hidden = num_hidden
        self.device = device
        self.embedding_h = nn.Linear(6, num_hidden)
        self.rnn = nn.RNN(int((num_input-6)/timestep), num_hidden, batch_first=True)
        self.fc = nn.Linear(num_hidden, num_output)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_h = x[:, 0:6]                                         # timestep = 0, AV state
        x = x[:, 6:].reshape((batch_size, self.timestep, -1))   # timestep = 1 ~ max_timestep, BV state
        h = self.embedding_h(x_h).unsqueeze(0)
        x, h = self.rnn(x, h)
        x = x[:, -1, :]
        output = self.fc(x)
        output = F.softmax(output, dim=1)[:,1]
        return output
