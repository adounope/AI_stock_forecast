import torch
import torch.nn as nn


class Basic(nn.Module):
    def __init__(self, history_size, forecast_size):
        super().__init__()
        self.sequent = nn.Sequential(
            nn.Linear(8*history_size, 8*history_size),
            nn.LeakyReLU(0.1),
            nn.Linear(8*history_size, 8*history_size),
            nn.LeakyReLU(0.1),
            nn.Linear(8*history_size, 4*forecast_size)
        )
    def forward(self, x):
        return self.sequent(x)


class Basic2(nn.Module):
    def __init__(self, history_size, forecast_size):
        super().__init__()
        self.sequent = nn.Sequential(
            nn.Linear(8*history_size, 8*history_size),
            nn.LeakyReLU(0.1),
            nn.Linear(8*history_size, 8*history_size),
            nn.LeakyReLU(0.1),
            nn.Linear(8*history_size, 1*forecast_size)
        )
    def forward(self, x):
        return self.sequent(x)
