import os, sys, glob
import argparse
import pandas as pd
import numpy as np
from funcs import printProgressBar
from featFuncs import loadFromJson
from DatasetHandling.DataClasses import Features, classToDic
import json

def main(featsList, jsonPath):
    """
    Adding mfcc features to a given dataset and writting its reference to a json file


    Example
    ----------
    python Standardize.py -f MFCC opensmile_ComParE_2016 MFB -j "/mnt/HD-Storage/Databases/RECOLA_46_P_S/data.json"
    python Standardize.py -f MFB -j "/mnt/HD-Storage/Datasets/Recola_46/data.json"
    python Standardize.py -f MFB -j "/mnt/HD-Storage/Datasets/AlloSat/data.json"
    python Standardize.py -f MFB -j "/mnt/HD-Storage/Datasets/Recola_46_S/data.json"
    """
    for featName in featsList:
        samples = loadFromJson(jsonPath)
        trainFilePaths = []
        print("feature:", featName)
        for i, ID in enumerate(samples.keys()):
            sample = samples[ID]
            featRef = sample["features"][featName]
            if sample["partition"] == "train": 
                path = featRef["path"]
                fullPath = os.path.join(os.path.split(jsonPath)[0], path)
                trainFilePaths.append(fullPath)
        mean, std = getMeanStd(trainFilePaths)

        for i, ID in enumerate(samples.keys()):
            sample = samples[ID]
            featRef = sample["features"][featName]
            newfeatName = featName+"_standardized"
            path = featRef["path"]
            fullPath = os.path.join(os.path.split(jsonPath)[0], path)
            fileOutPath = os.path.join(os.path.split(jsonPath)[0], path.replace(featName, newfeatName))
            standardize(fullPath, fileOutPath, mean, std)
            featsDict = getFeatsDict(newfeatName, featRef["genre"], featRef["dimension"], path.replace(featName, newfeatName))
            samples[ID]["features"][featsDict["ID"]] = featsDict
            printProgressBar(i + 1, len(samples), prefix = 'Standardizing '+ featName +' features', suffix = 'Complete', length = "fit")

        with open(jsonPath, 'w') as jsonFile:
            json.dump(samples, jsonFile,  indent=4, ensure_ascii=False)


def getFeatsDict(featName, featGenre, dim, path):
    feats = Features()
    feats.setParams(featName, featGenre, dim, path)
    return classToDic(feats)

def getMeanStd(filePaths):
    # filePaths = glob.glob(os.path.join(csvInFolder, "**", fileNamesWith+"*.csv"), recursive=True)
    df = pd.read_csv(filePaths[0])
    dims = len([aux for aux in df.keys() if "feat_" in aux])
    myFeats = [[] for _ in range(dims)]
    # print("dim", dim)
    for f, filePath in enumerate(filePaths):
        df = pd.read_csv(filePath)
        for dim in range(dims):
            keyWord = "feat_"+str(dim)
            col = list(df[keyWord])
            myFeats[dim] += col
        printProgressBar(f + 1, len(filePaths), prefix = 'Calculating Mean and Std:', suffix = 'Complete', length = "fit")

    allFeats = np.array(myFeats)
    mean = np.mean(allFeats, axis=1)
    std = np.std(allFeats, axis=1)
    return mean, std 

def inputReader(self, feat, jsonPath):
    # feats = self.dataPart[ID]["features"]
    # feat = feats[self.feat]
    path = feat["path"]
    dimension = feat["dimension"][-1]
    headers = ["feat_"+str(i) for i in range(dimension)]
    fullPath = os.path.join(os.path.split(jsonPath)[0], filePath)
    inputs = self.csvReader(fullPath, headers)
    return inputs

def csvReader(self, filePath, headers, standardize=False):
    # fullPath = os.path.join(self.DatasetsPath, filePath.replace("./", ""))
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

def standardize(fileInPath, fileOutPath, mean, std):
    """
    standardize csv feature files.

    """
    df = pd.read_csv(fileInPath)
    dims = len([aux for aux in df.keys() if "feat_" in aux])
    for dim in range(dims):
        keyWord = "feat_"+str(dim)
        col = df[keyWord].to_numpy()
        col = (col - mean[dim])/std[dim]
        df[keyWord] = col
    # outPath = filePath.replace(csvInFolder, csvOutFolder)
    dirName = os.path.dirname(fileOutPath)
    if not os.path.exists(dirName): os.makedirs(dirName)
    with open(fileOutPath, 'w', encoding='utf-8') as csvFile:
        df.to_csv(csvFile, index=False)
    # printProgressBar(f + 1, len(filePaths), prefix = 'Standardizing features:', suffix = 'Complete', length = "fit")


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--featsList', '-f', help="name of features", nargs='+', default=[])
    parser.add_argument('--json', '-j', help="json file path")
    args = parser.parse_args()
    Flag = False
    if args.featsList is None:  Flag=True
    if args.json is None:         Flag=True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.featsList, args.json)
    # wavPath = "../../Data/WavsProcessed"
    # csvPath = "../../Data/Feats_MFB"
    # calc_mfbs(wavPath, csvPath, rate=16000)
