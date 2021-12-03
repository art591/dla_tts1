import torch
from torch import nn


class ConvPermute(nn.Module):
    def __init__(self, embedding_size, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(embedding_size, embedding_size, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1)
        return y


class DurationPredictor(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.model = nn.Sequential(
            ConvPermute(embedding_size, kernel_size=3, padding='same'),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            ConvPermute(embedding_size, kernel_size=3, padding='same'),
            nn.LayerNorm(embedding_size),
            nn.ReLU()
        )
        self.output = nn.Linear(embedding_size, 1)

    def forward(self, x):
        y = self.model(x)
        return self.output(y)