import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class myTransformer(nn.Module):
    def __init__(self, featSize, hidden_size=128, nhead=8, num_layers=1, lastOnly=False, outputSize=1):
        super(myTransformer, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=featSize, nhead=nhead, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.transformer_encoder(x)
        if self.lastOnly: output = output[:, -1, :]
        output = self.linTh(output)
        return output


class mySincTransformer(nn.Module):
    def __init__(self, featSize, hidden_size=128, nhead=8, num_layers=1, fs=100, M=10, lastOnly=False, device="cuda", outputSize=1):
        super(mySincTransformer, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=featSize, nhead=nhead, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )
        self.sinc = SincModel(outputSize, fs=fs, M=M, device=device)

    def forward(self, x):
        output = self.transformer_encoder(x)
        if self.lastOnly: output = output[:, -1, :]
        output = self.linTh(x)
        output = self.sinc(output)
        return output


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
        

class myLinTanhSinc(nn.Module):
    def __init__(self, featSize, fs=100, M=10, lastOnly=False, device="cuda", outputSize=1):
        super(myLinTanhSinc, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        # self.sincIn = SincModel(featSize, fs=fs, M=M, device=device)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )
        self.sincOut = SincModel(outputSize, fs=fs, M=M, device=device)

    def forward(self, x):
        # output = self.sincIn(x)
        output = self.linTh(x)
        output = self.sincOut(output)
        return output


class mySincLinTanhSinc(nn.Module):
    def __init__(self, featSize, fs=100, M=10, lastOnly=False, device="cuda", outputSize=1):
        super(mySincLinTanhSinc, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.sincIn = SincModel(featSize, fs=fs, M=M, device=device)
        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )
        self.sincOut = SincModel(outputSize, fs=fs, M=M, device=device)

    def forward(self, x):
        output = self.sincIn(x)
        output = self.linTh(output)
        output = self.sincOut(output)
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


class SincModel(nn.Module):
    '''
        fc:  the cutoff frequency in Hz.
        fs:  the sampling frequency in Hz.
        M:   half of the signal in seconds. Total number of points is N = 2*M*fs + 1.
        tau: the time shift for the sinc kernel.
    '''
    def __init__(self, featSize, prefcs=None, fs=100, M=10, device="cuda", pretaus=None):
        super(SincModel, self).__init__()
        self.featSize = featSize
        self.device = device
        # self.kernel_size = kernel_size
        self.taus = pretaus if (not pretaus is None) else torch.zeros(featSize, device=self.device, requires_grad=False).float()
        # print(self.taus.size())
        self.prefcs = prefcs
        self.M = M
        self.fs = fs
        self.shift = self.M*self.fs
        self.sigmoid = nn.Sigmoid()
        self.sigWeight = 100
        self.startFc = fs/1000# -5 / self.sigWeight #startFc#fs/1000
        self.startFc = - np.log((1/self.startFc)-1) / self.sigWeight
        # print('self.startFc', self.startFc)
        self.fc = prefcs if (not prefcs is None) else torch.nn.Parameter(torch.zeros(1, device=self.device).float() + self.startFc, requires_grad=True)
        # self.fc = prefcs if (not prefcs is None) else torch.zeros(1, device=self.device, requires_grad=True).float() + self.startFc
        self.kernel_size = 2*M*fs + 1

    def getfc(self):
        if (not self.prefcs is None): 
            myfc=self.prefcs
        else:
            myfc = (1/(1+torch.exp(-self.sigWeight*self.fc))) * self.fs/2
        # myfc = (self.tanh(self.fc)/2 + 0.5) * self.fs/2
        # print(myfc)
        return myfc

    def MakeSincKernel(self, fc, fs, M, tau):
        N = 2*M*fs + 1 # total number of points (must be odd for symmetry)
        linspTorch1 = torch.linspace(-M, M, N, requires_grad=False, device=self.device)
        linspTorch = (linspTorch1 + tau * M)
        sinc1 = torch.sin(2 * np.pi * fc * linspTorch) / (np.pi * linspTorch)
        sinc1[torch.isnan(sinc1)] = 2*fc
        sinc = sinc1 / fs
        return sinc

    def sincForward(self, inputs):
        outputs = []
        # print(self.fs*self.fcs[0])
        myfc = self.getfc() # bounding fc to be between 0 and fs/2
        # print("myfc",myfc)
        for i in range(0, self.featSize):
            # print(self.fcWeight*self.fc)
            weight = self.MakeSincKernel(myfc, self.fs, self.M, self.taus[i])
            theInput = inputs[:, :, :, i].view(inputs.size()[0], inputs.size()[1], inputs.size()[2], 1)
            output = torch.nn.functional.conv2d(theInput, weight.view(1, 1, self.kernel_size, 1))
            # from matplotlib import pyplot as plt
            # size = output.size()[2]
            # X = list(range(size))
            # plt.plot(X[:-self.shift],theInput.view(theInput.size()[2])[self.kernel_size-1:-self.shift].cpu().data.numpy(), color='red')
            # plt.plot(X[:-self.shift],output.view(output.size()[2])[self.shift:].cpu().data.numpy(), color='blue')
            # plt.legend(['ins', 'outs'], loc=6)
            # plt.show()
            outputs.append(output)#output[:, :, self.shift:, :]
        return torch.cat(outputs, dim=3)

    def forward(self, x):
        batch_size, timesteps, sq_len = x.size()
        x = x.view(batch_size, 1, timesteps, sq_len) # if sq_len more than one, it'll be used as input_channel
        pad = torch.nn.ConstantPad2d((0, 0, self.shift, self.shift), 0) # add padding to equalize size
        x = pad(x)
        output = self.sincForward(x)
        output = output.squeeze(1)
        return output


