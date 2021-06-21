import os, sys, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import argparse
import pandas as pd
import numpy as np
from Utils.Funcs import printProgressBar, loadFromJson
from DataClasses import Annotation, classToDic
import json
import random

def main(partitions, jsonPath, basedOnFolder, trainIdsForced, devIdsForced, testIdsForced, outJson):
    """
    Repartitioning files in a json file


    Example
    ----------
    python repartitionJson.py -p 60 20 20 -f True -j "/mnt/HD-Storage/Datasets/RAVDESS/data.json"
    python repartitionJson.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_parted.json" -tr user_2 user_3 user_4 user_5 user_7 user_8 user_9 user_10 user_11 -de user_12 user_13 user_14 user_15 -te user_16 user_17 user_18 user_19
    python repartitionJson.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -tr user_2 user_3 user_4 user_5 user_7 user_8 user_9 user_10 user_11 -de user_12 user_13 user_14 user_15 -te user_16 user_17 user_18 user_19
    
    python repartitionJson.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted.json" -tr Ses01 Ses02 Ses03 -de Ses04 -te Ses05

    python repartitionJson.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted.json" -tr B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 -de B14 B15 B16 B17 B18 B19 B20 -te B21 B22 B23 B24 B25 B26 B27
    """
    samples = loadFromJson(jsonPath)
    ids = list(samples.keys())
    if basedOnFolder:
        folders = []
        foldersIds = {}
        for i, ID in enumerate(samples.keys()):
            sample = samples[ID]
            wavePath = sample["path"]
            folder = os.path.split(wavePath)[0]
            if not folder in folders: 
                folders.append(folder)
                foldersIds[folder] = [ID]
            else:
                foldersIds[folder].append(ID)
        ids = folders
    # print(folders)
    if len(partitions) != 0:
        total = len(ids)
        random.shuffle(ids)
        trainCut = int(total*partitions[0]/100)
        devCut = int(total*(partitions[0]+partitions[1])/100)
        trainIds = ids[:trainCut]
        devIds = ids[trainCut:devCut]
        testIds = ids[devCut:]

    if trainIdsForced != []:
        trainIds = []
        devIds = []
        testIds = []
        for ID in ids:
            trainFlag = False
            devFlag = False
            testFlag = False
            for trainId in trainIdsForced:
                if trainId in ID: trainFlag = True; break
            for devId in devIdsForced:
                if devId in ID: devFlag = True; break
            for testId in testIdsForced:
                if testId in ID: testFlag = True; break
            if trainFlag: trainIds.append(ID)
            if devFlag: devIds.append(ID)
            if testFlag: testIds.append(ID)
    # print(trainIds)

    for i, idx in enumerate(ids):
        partition = "train"
        if idx in devIds:  partition = "dev"
        if idx in testIds: partition = "test"
        if basedOnFolder:
            for eachIdx in foldersIds[idx]:
                samples[eachIdx]["partition"] = partition
        else:
            samples[idx]["partition"] = partition
        printProgressBar(i + 1, len(ids), prefix = 'Repartitioning ', suffix = 'Complete', length = "fit")

    if outJson == "": outJson=jsonPath
    directory = os.path.dirname(outJson)
    if not os.path.exists(directory): os.makedirs(directory)
    with open(outJson, 'w') as jsonFile:
            json.dump(samples, jsonFile,  indent=4, ensure_ascii=False)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partitions', '-p', help="percentage of each partition", nargs='+', default=[60, 20, 20])
    parser.add_argument('--basedOnFolder', '-f', help="if True, would put all files in the same folder into the same partition", default=False)
    parser.add_argument('--json', '-j', help="json file path")
    parser.add_argument('--trainIds', '-tr', help="force training data to have the given strings in their IDs, would disregard partition percentage!", nargs='+', default=[])
    parser.add_argument('--devIds', '-de', help="force development data to have the given strings in their IDs", nargs='+', default=[])
    parser.add_argument('--testIds', '-te', help="force testing data to have the given strings in their IDs", nargs='+', default=[])
    parser.add_argument('--outJson', '-o', help="if not given, would overwrite the input json!", default="")
    args = parser.parse_args()
    Flag = False
    args.partitions = [float(part) for part in args.partitions]
    if len(args.trainIds)!=0:
        args.partitions = []
    else:
        if sum(args.partitions) != 100: Flag=True; print("the sum of the three partition percentage should add up to 100!")
    if args.json is None:           Flag=True
    if args.basedOnFolder == "False": args.basedOnFolder = False
    if args.basedOnFolder == "True":  args.basedOnFolder = True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.partitions, args.json, args.basedOnFolder, args.trainIds, args.devIds, args.testIds, args.outJson)