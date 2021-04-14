import os, glob
import pandas as pd
import numpy as np
import json

class DataReader():
    '''
        Read Data Set based on a json file.
    '''
    def __init__(self, jsonPath, targetFunc=None):
        super(DataReader, self).__init__()
        self.jsonPath = jsonPath
        self.DatasetsPath = os.path.dirname(jsonPath)
        self.dataReaderType = "Classic" # Allows different types of reading data (through a setter function) for different intended purpusos, e.g. for classical train-dev-test for feat->annot, or end2end learning (wav->annot), autoencoders (feat->feat), or other things like feat1->feat2, ...
        self.targetFunc = targetFunc
        with open(jsonPath, 'r') as jsonFile: 
            self.data = json.load(jsonFile)

    def getPartition(self, partition):
        dataPart = {}
        for ID, sample in self.data.items():
            if sample["partition"] == partition: 
                dataPart[ID] = sample
        return dataPart
    
    def setDatasetClassic(self, partition, feat, annot): 
        # put annot as None if no annot -> target will be the same as feat then, or we can define other formats with other kinds of setDataset funcs
        self.dataReaderType = "Classic"
        self.dataPart = self.getPartition(partition)
        self.feat = feat
        self.annot = annot

    def setDatasetFeatOnly(self, partition, feat): 
        # put annot as None if no annot -> target will be the same as feat then, or we can define other formats with other kinds of setDataset funcs
        self.dataReaderType = "FeatOnly"
        self.dataPart = self.getPartition(partition)
        self.feat = feat

    def inputReader(self, ID):
        feats = self.dataPart[ID]["features"]
        feat = feats[self.feat]
        path = feat["path"]
        dimension = feat["dimension"][-1]
        headers = ["feat_"+str(i) for i in range(dimension)]
        inputs = self.csvReader(path, headers)
        return inputs

    def targetReader(self, ID):
        annots = self.dataPart[ID]["annotations"]
        annot = annots[self.annot]
        path = annot["path"]
        headers = annot["headers"]
        targets = self.csvReader(path, headers)
        if not self.targetFunc is None: targets = self.targetFunc(targets)
        return targets
    
    def csvReader(self, filePath, headers, standardize=False):
        fullPath = os.path.join(self.DatasetsPath, filePath.replace("./", ""))
        # print(fullPath)
        df = pd.read_csv(fullPath)
        outs = []
        for header in headers:
            out = df[header].to_numpy()
            if standardize: out = (out - out.mean(axis=0)) / out.std(axis=0)
            out = np.expand_dims(out, axis=1)
            outs.append(out)
        outs = np.concatenate(outs, 1)
        return outs

    @staticmethod
    def getMeanStd(vector):
        # print("vector.shape",vector.shape)
        mean = np.mean(vector, 0)
        std  = np.std(vector, 0)
        out = np.array([mean, std]).reshape(vector.shape[1], -1)
        # print("out.shape",out.shape)
        return out

    def setTargetMeanStd(self):
        self.targetFunc = DataReader.getMeanStd

    def __len__(self):
        return len(self.dataPart)

    def __getitem__(self, idx):
        ID = list(self.dataPart.keys())[idx]
        if self.dataReaderType == "Classic":
            inputs = self.inputReader(ID)
            targets = self.targetReader(ID)
            return inputs, targets
        elif self.dataReaderType == "FeatOnly":
            inputs = self.inputReader(ID)
            return ID, inputs
        else:
            print("Please set the dataset first")
    

# dataset = DataReader("/Users/sinaalisamir/Documents/Datasets/RECOLA_46_P/data.json")
# dataset.setDatasetClassic("dev", "MFB", "gs_arousal_0.01")
# feat, tar = dataset[0]
# print(feat.shape, tar.shape)
