import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class myLinTanh(nn.Module):
    def __init__(self, featSize, lastOnly=False, device="cuda", outputSize=1):
        super(myLinTanh, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        # self.sincIn = SincModel(featSize, fs=fs, M=M, device=device)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # output = self.sincIn(x)
        output = self.linTh(x)
        return output
        

class myLinLinTanh(nn.Module):
    def __init__(self, featSize, hidden_size=128, lastOnly=False, device="cuda", outputSize=1):
        super(myLinLinTanh, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        # self.sincIn = SincModel(featSize, fs=fs, M=M, device=device)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # output = self.sincIn(x)
        output = self.linTh(x)
        return output


class myLinLinTanhSinc(nn.Module):
    def __init__(self, featSize, hidden_size=128, fs=100, M=10, lastOnly=False, device="cuda", outputSize=1):
        super(myLinLinTanhSinc, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        # self.sincIn = SincModel(featSize, fs=fs, M=M, device=device)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outputSize),
            nn.Tanh(),
        )
        self.sincOut = SincModel(outputSize, fs=fs, M=M, device=device)

    def forward(self, x):
        # output = self.sincIn(x)
        output = self.linTh(x)
        output = self.sincOut(output)
        return output


class myGRU(nn.Module):
    def __init__(self, featSize, hidden_size=128, num_layers=1, device="cuda", lastOnly=False, outputSize=1):
        super(myGRU, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.linTh = nn.Sequential(
            nn.Linear(hidden_size, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        if self.lastOnly: output = output[:, -1, :].unsqueeze(1)
        output = self.linTh(output)
        return output


class myGRUFF(nn.Module):
    def __init__(self, featSize, hidden_size=256, num_layers=1, FF_hidden_size=128, device="cuda", lastOnly=False, outputSize=1):
        super(myGRUFF, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.linTh = nn.Sequential(
            nn.Linear(hidden_size, FF_hidden_size),
            nn.ReLU(),
            nn.Linear(FF_hidden_size, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        if self.lastOnly: output = output[:, -1, :]
        output = self.linTh(output)
        return output


class myFFGRUFF(nn.Module):
    def __init__(self, featSize, hidden_size=256, num_layers=1, FF_hidden_size=128, device="cuda", lastOnly=False, outputSize=1):
        super(myFFGRUFF, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.linThIn = nn.Sequential(
            nn.Linear(featSize, FF_hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=FF_hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.linTh = nn.Sequential(
            nn.Linear(hidden_size, FF_hidden_size),
            nn.ReLU(),
            nn.Linear(FF_hidden_size, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output = self.linThIn(x)
        output, _ = self.rnn(output)
        if self.lastOnly: output = output[:, -1, :]
        output = self.linTh(output)
        return output

