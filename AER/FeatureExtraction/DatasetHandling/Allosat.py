from DataClasses import *
import os, glob
import pandas as pd
import json
from funcs import printProgressBar

def main():
    Datasets_Path = "/mnt/HD-Storage/Datasets"
    main_path     = os.path.join(Datasets_Path, "AlloSat")
    wavs_path     = os.path.join(main_path, "Wavs")
    jsonPath      = os.path.join(main_path, "data.json")

    path = os.path.join(wavs_path, "**", "*.wav")
    filesPaths = glob.glob(path, recursive=True)

    allDics = {}

    trainList, devList, testList = getParts(os.path.join(main_path, "Parts"))

    for i, filePath in enumerate(filesPaths):
        baseName = os.path.basename(filePath)[:-4]
        fileDict = AudioSample()

        partition = ""
        if baseName in trainList: partition = "train"
        if baseName in devList:   partition = "dev"
        if baseName in testList:  partition = "test"

        fileDict.setParams(baseName, filePath, partition)
        # fileDict.features = getFeatures(main_path, baseName)
        fileDict.annotations = getAnnots(main_path, baseName)
        fileDict.speaker_info.language = "French"

        dic = classToDic(fileDict.__dict__)
        # dic = changePaths(dic, main_path, ".")
        dic = localizePaths(dic, main_path)
        allDics[baseName] = dic
        printProgressBar(i + 1, len(filesPaths), prefix = 'Processing Files:', suffix = 'Complete')

    with open(jsonPath, 'w') as fp:
        json.dump(allDics, fp,  indent=4, ensure_ascii=False)

def getParts(path):
	trainListPath = os.path.join(path, "train.txt")
	devListPath   = os.path.join(path, "dev.txt"  )
	testListPath  = os.path.join(path, "test.txt" )
	trainList = pd.read_csv(trainListPath, delimiter = "\t", header=None).to_numpy()
	trainList = trainList.reshape(trainList.shape[0])
	devList = pd.read_csv(devListPath, delimiter = "\t", header=None).to_numpy()
	devList = devList.reshape(devList.shape[0])
	testList = pd.read_csv(testListPath, delimiter = "\t", header=None).to_numpy()
	testList = testList.reshape(testList.shape[0])
	return trainList, devList, testList
	

def getAnnots(main_path, baseName):
    annots_path = "Annots"
    path = os.path.join(main_path, annots_path)
    IDs = ["labels_0.25", "labels_0.02", "labels_0.01"]
    annots = {}
    for ID in IDs:
        for genre in ["satisfaction"]:
            annots[ID] = classToDic(getAnnotsBy(path, ID, genre, baseName))
    return annots

def getAnnotsBy(path, folder, genre, baseName):
    path = os.path.join(path, folder, baseName+".csv")
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)
    annots = Annotation()
    dimension = df.shape
    headers = [genre]
    annots.setParams(folder, genre, dimension, path, headers)
    annots.annotator_info.language = "French"
    return annots

if __name__== "__main__":
    main()


