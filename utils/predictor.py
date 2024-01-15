import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class predictor_dnn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=256, output_dim=2, device='cuda', dropout=False) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if isinstance(x, np.ndarray):
        #     x = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        if self.dropout:
            x = self.dropout_layer(self.fc1(x))
            x = F.relu(x)
            x = self.dropout_layer(self.fc2(x))
            x = F.relu(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
        output = self.fc3(x)
        
        # output = 2 * torch.tanh(x)  # soft clip
        if self.output_dim == 1:
            output = torch.sigmoid(output).squeeze()
        elif self.output_dim == 2:
            output = F.softmax(output, dim=1)[:,-1]
        return output   # probability of predicting positive samples; shape: [batch_size, 1]


class predictor_rnn(nn.Module):
    def __init__(self, timestep: int, input_dim: int, hidden_dim=256, output_dim=2, device='cuda', dropout=False) -> None:
        super().__init__()
        self.timestep = timestep
        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = dropout
        self.embedding_h = nn.Linear(6, hidden_dim)
        self.rnn = nn.RNN(int((input_dim-6)/timestep), hidden_dim, batch_first=True, dropout=0.2 if self.dropout else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_h = x[:, 0:6]                                         # timestep = 0, AV state; shape: [batch_size, 6]
        x = x[:, 6:].reshape((batch_size, self.timestep, -1))   # timestep = 0 ~ max_timestep, BV state; shape: [batch_size, timestep, max_bv_num*6]
        h = self.embedding_h(x_h).unsqueeze(0)  # shape: [1, batch_size, hidden_dim]
        x, h = self.rnn(x, h)
        x = x[:, -1, :]
        output = self.fc(x)
        output = F.softmax(output, dim=1)[:,1]
        return output


class predictor_vae(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=(256,64,16), latent_dim=2, device='cuda') -> None:
        super().__init__()
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], input_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        output = self.classifier(z).squeeze()
        return decoded, output, mu, logvar
