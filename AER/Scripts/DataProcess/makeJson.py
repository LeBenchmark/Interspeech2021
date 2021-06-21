import os, sys, glob, argparse
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from DataProcess.DataClasses import *
import pandas as pd
import json
from Utils.Funcs import printProgressBar

def main(mainPath, partitions):    
    """
        Make the json file only given the wavs files of a dataset.

        example: 
            python makeJson.py -m "/mnt/HD-Storage/Datasets/ESTER1" -p "train/*.wav" "dev/*.wav" "test/*.wav"
            python makeJson.py -m "/mnt/HD-Storage/Datasets/RAVDESS"

            python makeJson.py -m "/mnt/HD-Storage/Datasets/SEMAINE"
            python makeJson.py -m "/mnt/HD-Storage/Datasets/IEMOCAP"

            python makeJson.py -m "/mnt/HD-Storage/Datasets/MaSS_Fr"

    """
    
    jsonPath = os.path.join(mainPath, "data.json")
    wavsPath = os.path.join(mainPath, "Wavs")
    if partitions == []:
        trainPath = os.path.join(wavsPath, "**","*.wav")
    else:
        trainPath = os.path.join(wavsPath, partitions[0])
        print("trainPath", trainPath)
    trainFiles = glob.glob(trainPath, recursive=True)
    print("trainFiles", trainFiles)
    if partitions == []:
        devFiles, testFiles = [], []
    else:
        devPath = os.path.join(wavsPath, partitions[1])
        devFiles = glob.glob(devPath, recursive=True)
        testPath = os.path.join(wavsPath, partitions[2])
        testFiles = glob.glob(testPath, recursive=True)

    allDics = {}

    for f, filesPaths in enumerate([trainFiles, devFiles, testFiles]):
        partition = ""
        if f == 0: partition = "train"
        if f == 1: partition = "dev"
        if f == 2: partition = "test"
        for i, filePath in enumerate(filesPaths):
            baseName = os.path.basename(filePath)[:-4]
            fileDict = AudioSample()

            fileDict.setParams(baseName, filePath, partition)
            # fileDict.features = getFeatures(main_path, baseName)

            dic = classToDic(fileDict.__dict__)
            # dic = changePaths(dic, main_path, ".")
            dic = localizePaths(dic, mainPath)
            allDics[baseName] = dic
            printProgressBar(i + 1, len(filesPaths), prefix = 'Processing Files for ' + partition + ":", suffix = 'Complete')

    with open(jsonPath, 'w') as fp:
        json.dump(allDics, fp,  indent=4, ensure_ascii=False)

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mainPath', '-m', help="where all the wav files are stored")
    parser.add_argument('--partitions', '-p', help="how the data are partitioned", nargs='+', default=[])
    args = parser.parse_args()
    Flag = False
    if args.mainPath   is None:    Flag=True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.mainPath, args.partitions)