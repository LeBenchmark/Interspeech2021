import os, sys, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import argparse
import pandas as pd
import numpy as np
from Utils.Funcs import printProgressBar, loadFromJson
from DataProcess.DataClasses import Annotation, classToDic
import json

def main(annotsList, headers, genres, jsonPath):
    """
    Adding annotations to a given dataset and writting its reference to a json file


    Example
    ----------
    python addAnnots.py -a "gs_arousal_0.01_std" "gs_valence_0.01_std" "gen_gs_arousal_0.01_std" "gen_gs_valence_0.01_std" -d "GoldStandard" -g arousal valence arousal valence -j "/mnt/HD-Storage/Datasets/Recola_46_S/data.json"
    python addAnnots.py -a "VAD_0.01" -d "label" -g "VAD" -j "/mnt/HD-Storage/Datasets/ESTER1/data.json"
    python addAnnots.py -a "turns_0.01" -d "label" -g "VAD" -j "/mnt/HD-Storage/Datasets/Recola_46/data.json"
    python addAnnots.py -a "turns_0.01" -d "label" -g "VAD" -j "/mnt/HD-Storage/Datasets/SEMAINE/data.json"
    python addAnnots.py -a "turns_0.01" -d "label" -g "VAD" -j "/mnt/HD-Storage/Datasets/IEMOCAP/data.json"
    python addAnnots.py -a "turns_0.01" -d "label" -g "VAD" -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data.json"
    """
    samples = loadFromJson(jsonPath)
    for t, annotName in enumerate(annotsList):
        trainFilePaths = []
        print("annot:", annotName)

        for i, ID in enumerate(samples.keys()):
            sample = samples[ID]
            wavePath = sample["path"]
            wavsFolder = wavePath.split(os.sep)[0]
            waveFullPath = os.path.join(os.path.split(jsonPath)[0], wavePath)
            featsLocalPath = wavePath.replace(wavsFolder, annotName).replace(".wav", ".csv")
            featsLocalPath = os.path.join("Annots", featsLocalPath)
            featsFullPath  = os.path.join(os.path.split(jsonPath)[0], featsLocalPath)

            try:
                df = pd.read_csv(featsFullPath, delimiter=',')
                out = df[headers].to_numpy().astype('float64')
                dim = list(out.shape)
                annotsDict = getAnnotsDict(annotName, genres[t], dim, featsLocalPath, headers)
                samples[ID]["annotations"][annotsDict["ID"]] = annotsDict
            except:
                print("Warning: could not read", featsFullPath)

            printProgressBar(i + 1, len(samples), prefix = 'Adding '+ annotName +' annotation', suffix = 'Complete', length = "fit")

        with open(jsonPath, 'w') as jsonFile:
            json.dump(samples, jsonFile,  indent=4, ensure_ascii=False)


def getAnnotsDict(annotName, annotGenre, dim, path, headers):
    annot = Annotation()
    annot.setParams(annotName, annotGenre, dim, path, headers)
    return classToDic(annot)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotsList', '-a', help="name of annots", nargs='+', default=[])
    parser.add_argument('--headers', '-d', help="name of headers", nargs='+', default=[])
    parser.add_argument('--genres', '-g', help="name of genres", nargs='+', default=[])
    parser.add_argument('--json', '-j', help="json file path")
    args = parser.parse_args()
    Flag = False
    if args.annotsList is None:   Flag=True
    if args.json is None:         Flag=True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.annotsList, args.headers, args.genres, args.json)
    # wavPath = "../../Data/WavsProcessed"
    # csvPath = "../../Data/Feats_MFB"
    # calc_mfbs(wavPath, csvPath, rate=16000)
