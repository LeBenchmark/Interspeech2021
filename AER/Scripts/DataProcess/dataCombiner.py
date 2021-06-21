import os, sys, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import argparse
import pandas as pd
import numpy as np
from Utils.Funcs import printProgressBar, loadFromJson
from DataClasses import Annotation, classToDic
import json
import array
import math
import random
import wave

def main(jsonPaths, outJson):
    """
    given a list of data.json files, combine all the data and put them all in one json file


    Example
    ----------
    python dataCombiner.py -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" "/mnt/HD-Storage/Datasets/MaSS_Fr/data.json" -o "/mnt/HD-Storage/Datasets/Mixed/data_RecolaMaSS.json"

    """
    dirname = os.path.dirname(outJson)
    if not os.path.exists(dirname): os.makedirs(dirname)

    newSamples = {}
    for jsonPath in jsonPaths:
        # print(jsonPath)
        # print(os.path.dirname(os.path.relpath(jsonPath, outJson)))
        addPath = os.path.dirname(os.path.relpath(jsonPath, outJson))
        samples = loadFromJson(jsonPath)
        for sample in samples:
            # print("before",samples[sample])
            addToPaths(samples[sample], addPath=addPath)
            # print("after",samples[sample])
            newSamples[sample] = samples[sample]
    with open(outJson, 'w') as jsonFile:
        json.dump(newSamples, jsonFile,  indent=4, ensure_ascii=False)

def addToPaths(DicItem, addPath="", key="path"):
    for item in DicItem:
        if item == key:
            DicItem[item] = os.path.join(addPath, DicItem[item])
        elif isinstance(DicItem[item], dict):
            addToPaths(DicItem[item], addPath=addPath, key=key)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="input json files path", nargs='+', default=[])
    parser.add_argument('--outJson', '-o', help="output json file path")

    args = parser.parse_args()
    Flag = False
    if len(args.json)==0:    Flag=True
    if args.outJson is None: Flag=True

    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.json, args.outJson)

