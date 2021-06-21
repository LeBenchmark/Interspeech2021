# from Dataset import Dataset
from torch.utils.data import DataLoader
from Utils.Funcs import printProgressBar, writeLineToCSV
from DataProcess.DataReader import DataReader
import os, glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import numbers
import copy
from Torch_Models.Models import *
from Torch_Models.Criterions import *

class ModelWrapper():
    def __init__(self, savePath="./wrapperDumps", device="cpu", seed=0, cutToEqualize=True, printLvl=10):
        super(ModelWrapper, self).__init__()
        '''
            Training, logs, analytics, and everything that can be used to easily train, test and anylyse pytorch models.
        '''
        self.savePath = savePath
        if not os.path.exists(self.savePath): os.makedirs(self.savePath)
        self.printLvl = printLvl
        self.device = device
        self.currentEpoch = 1
        self.cutToEqualize = cutToEqualize # cut targets to equalize the outputs and targets sizes
        self.noMoreTrain = False # Flag for early stopping.
        self.modelStates = {} # Different model states in different epochs.
        self.epochDevLosses = [] # Different epoch losses for the develpment set.
        self.epochTimes = [] # Keeping how much time has passed for each training epoch.
        self.classification = False
        
        self.checkPointPath = os.path.join(self.savePath, "checkpoint.pth")
        self.csvLogPath = os.path.join(self.savePath, "log.csv")

        torch.manual_seed(seed)
        if device=="cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setModel(self, modelName, inputDim, outputDim, paramsStr): # the model is referenced!
        self.modelName = modelName
        if "CLS" in self.modelName: self.classification = True
        if paramsStr != "":
            params = paramsStr.split("-")
            params = [int(param) for param in params]
        if modelName == "Transformer":
            self.model = myTransformer(inputDim, hidden_size=params[0], nhead=params[1], num_layers=params[2], outputSize=outputDim)
        if modelName == "GRU_CLS":
            self.model = myGRU_CLS(inputDim, hidden_size=params[0], num_layers=params[1], outputSize=outputDim, lastOnly=True)
        if modelName == "GRU":
            self.model = myGRU(inputDim, hidden_size=params[0], num_layers=params[1], outputSize=outputDim)
        if modelName == "GRULast":
            self.model = myGRU(inputDim, hidden_size=params[0], num_layers=params[1], outputSize=outputDim, lastOnly=True)
        if modelName == "GRUFF":
            self.model = myGRUFF(inputDim, hidden_size=params[0], num_layers=params[1], FF_hidden_size=params[2], outputSize=outputDim)
        if modelName == "FFGRUFF":
            self.model = myFFGRUFF(inputDim, hidden_size=params[0], num_layers=params[1], FF_hidden_size=params[2], outputSize=outputDim)
        if modelName == "SincTransformer":
            self.model = mySincTransformer(inputDim, hidden_size=params[0], nhead=params[1], num_layers=params[2], fs=params[3], M=params[4], outputSize=outputDim, device=self.device)
        if modelName == "LinTanhSinc":
            self.model = myLinTanhSinc(inputDim, fs=params[0], M=params[1], outputSize=outputDim, device=self.device)
        if modelName == "SincLinTanhSinc":
            self.model = mySincLinTanhSinc(inputDim, fs=params[0], M=params[1], outputSize=outputDim, device=self.device)
        if modelName == "LinLinTanhSinc":
            self.model = myLinLinTanhSinc(inputDim, hidden_size=params[0], fs=params[1], M=params[2], outputSize=outputDim, device=self.device)
        if modelName == "LinLinTanh":
            self.model = myLinLinTanh(inputDim, hidden_size=params[0], outputSize=outputDim, device=self.device)
        if modelName == "LinTanh":
            self.model = myLinTanh(inputDim, outputSize=outputDim, device=self.device)
        self.model.to(device=self.device)

    def setOptimizer(self, optimizerName, learningRate=0.001):
        optimizers = {
            "Adam": torch.optim.Adam(self.model.parameters(), lr=learningRate)
        }
        self.optimizer = optimizers[optimizerName]

    def setCriterion(self, criterionName):
        criterions = {
            "CCC": CCC_Loss(),
            "MSELoss": torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean'),
            "CELoss": torch.nn.CrossEntropyLoss()
        }
        self.criterion = criterions[criterionName]

    def trainModel(self, datasetTrain, datasetDev, batchSize=1, maxEpoch=200, loadBefore=True, 
                    tolerance = 15, minForTolerance=15, limitTrainData=False, limitDevData=False):
        if loadBefore: self.loadCheckpoint()
        trainDataloader = DataLoader(dataset=datasetTrain, batch_size=batchSize, shuffle=True)
        devDataloader   = DataLoader(dataset=datasetDev,   batch_size=batchSize, shuffle=False)
        while self.currentEpoch <= maxEpoch:
            if self.noMoreTrain: 
                if self.printLvl>0: print("Early stopping has been achieved!")
                break
            if limitTrainData:
                datasetTrain.limitData(limitTrainData)
                trainDataloader = DataLoader(dataset=datasetTrain, batch_size=batchSize, shuffle=True)
            if limitDevData:
                datasetDev.limitData(limitTrainData)
                devDataloader = DataLoader(dataset=datasetDev, batch_size=batchSize, shuffle=False)
            self.trainEpoch(trainDataloader)
            devLoss = self.evaluateModel(devDataloader)
            self.modelStates[self.currentEpoch] = copy.deepcopy(self.model.state_dict())
            self.epochDevLosses.append(devLoss)
            if self.printLvl>1: 
                printProgressBar(self.currentEpoch, maxEpoch, prefix = 'Training model:', suffix = '| epoch loss: '+str(devLoss), length = "fit")
                # print("loss", self.currentEpoch, devLoss)
            self.currentEpoch += 1
            # --- Early Stopping ---
            if (self.currentEpoch - self.getBestEpochIdx() >= tolerance) and self.currentEpoch > minForTolerance:
                self.noMoreTrain = True 
            self.saveCheckpoint()
        self.saveLogToCSV()
        if self.printLvl>0: print("Training the model has been finished!")

    def trainEpoch(self, trainDataloader):
        timeStart = time.time()
        for (inputs, targets) in trainDataloader:
            self.trainStep(inputs, targets)
        timeElapsed = time.time() - timeStart
        self.epochTimes.append(timeElapsed)

    def evaluateModel(self, dataloader):
        self.model.eval()
        lossSum = 0
        dataAmount = 0
        for (inputs, targets) in dataloader:
            loss = self.calcLoss(inputs, targets)
            batchSize = inputs.size()[0]
            dataAmount += batchSize
            lossSum += loss.detach().cpu().numpy() * batchSize
        lossMean = lossSum / dataAmount
        return lossMean

    def calcLoss(self, theInput, theTarget):
        targets = Variable(theTarget.float())
        if self.classification: targets = Variable(theTarget.long()).squeeze(2).squeeze(1)
        targets = targets.to(device=self.device)
        # print(theInput.size(), targets.size())
        outputs = self.forwardModel(theInput)
        # print("sizes", outputs.size(), targets.size())
        if self.cutToEqualize and (not self.classification):  # equalising sizes -----------------
            if outputs.size()[1] < targets.size()[1]:
                targets = targets[:,:outputs.size()[1],:]
            elif outputs.size()[1] > targets.size()[1]:
                outputs = outputs[:,:targets.size()[1],:]
        # print(targets.size(), outputs.size())
        # print("outputs", outputs)
        # print("targets", targets)
        loss = self.criterion(outputs, targets)
        return loss

    def trainStep(self, theInput, theTarget):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.calcLoss(theInput, theTarget)
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def forwardModel(self, theInput):
        inputs = Variable(theInput.float())
        inputs = inputs.to(device=self.device)
        outputs = self.model(inputs)
        return outputs

    def getBestEpochIdx(self):
        return self.epochDevLosses.index(min(self.epochDevLosses))+1

    def saveLogToCSV(self):
        csvPath = self.csvLogPath
        headers = ["epoch", "evalLossMean", "trainTime"]
        epochs = list(range(len(self.epochDevLosses)))
        data = np.array([epochs, self.epochDevLosses, self.epochTimes])
        data[0] += 1 # so that epochs start at one not zero! :)
        data = np.transpose(data)
        df = pd.DataFrame(data, columns = headers)
        df.to_csv(csvPath, index=False)

    def loadBestModel(self):
        checkpoint = torch.load(self.checkPointPath, map_location=self.device)
        # print(self.epochDevLosses, self.epochDevLosses.index(min(self.epochDevLosses)), len(self.epochDevLosses))
        # print(checkpoint['modelStates'])
        bestEpoch = self.epochDevLosses.index(min(self.epochDevLosses))+1
        self.model.load_state_dict(checkpoint['modelStates'][bestEpoch])

    def loadCheckpoint(self):
        try:
            checkpoint = torch.load(self.checkPointPath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.currentEpoch = checkpoint['currentEpoch']
            self.modelStates = checkpoint['modelStates']
            self.epochDevLosses = checkpoint['epochDevLosses']
            self.epochTimes = checkpoint['epochTimes']
            self.noMoreTrain = checkpoint['noMoreTrain']
            # self.model = torch.load(self.saveFilePath, map_location=self.device)
            # print(self.epochDevLosses)
        except:
            if os.path.exists(self.checkPointPath):
                if self.printLvl>0: print("Warning: Could not load previous model, if there was any, it will get overwritten!")

    def saveCheckpoint(self):
        # torch.save(self.model, self.saveFilePath)
        checkpoint = {
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'currentEpoch': self.currentEpoch,
            'modelStates': self.modelStates, 
            'epochDevLosses': self.epochDevLosses, 
            'epochTimes': self.epochTimes,
            'noMoreTrain': self.noMoreTrain, # Saving this to avoid training accidentaly later after reaching early stopping!
        }
        torch.save(checkpoint, self.checkPointPath)





