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

def main(jsonPath, outJson, percentageReduction, keepTest, blackList, whiteList):
    """
    Reduce data in a json file to have a smaller subset of the original dataset


    Example
    ----------
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_short_10.json" -p 10

    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr5_whiteNoise.json" -w snr5_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr5_Ads_2018_2020.json" -w snr5_trainNoise_Ads_2018_2020 snr5_devNoise_Ads_2018_2020 snr5_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr5_News.json" -w snr5_trainNoise_News snr5_devNoise_News snr5_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr5_TalkShows.json" -w snr5_trainNoise_TalkShows snr5_devNoise_TalkShows snr5_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr5_Ambient_Music_top_charts.json" -w snr5_trainNoise_Ambient_Music_top_charts snr5_devNoise_Ambient_Music_top_charts snr5_testNoise_Ambient_Music_top_charts && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr15_whiteNoise.json" -w snr15_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr15_Ads_2018_2020.json" -w snr15_trainNoise_Ads_2018_2020 snr15_devNoise_Ads_2018_2020 snr15_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr15_News.json" -w snr15_trainNoise_News snr15_devNoise_News snr15_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr15_TalkShows.json" -w snr15_trainNoise_TalkShows snr15_devNoise_TalkShows snr15_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy_snr15_Ambient_Music_top_charts.json" -w snr15_trainNoise_Ambient_Music_top_charts snr15_devNoise_Ambient_Music_top_charts snr15_testNoise_Ambient_Music_top_charts
    
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr5_whiteNoise.json" -w snr5_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr5_Ads_2018_2020.json" -w snr5_trainNoise_Ads_2018_2020 snr5_devNoise_Ads_2018_2020 snr5_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr5_News.json" -w snr5_trainNoise_News snr5_devNoise_News snr5_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr5_TalkShows.json" -w snr5_trainNoise_TalkShows snr5_devNoise_TalkShows snr5_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr5_Ambient_Music_top_charts.json" -w snr5_trainNoise_Ambient_Music_top_charts snr5_devNoise_Ambient_Music_top_charts snr5_testNoise_Ambient_Music_top_charts && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr15_whiteNoise.json" -w snr15_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr15_Ads_2018_2020.json" -w snr15_trainNoise_Ads_2018_2020 snr15_devNoise_Ads_2018_2020 snr15_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr15_News.json" -w snr15_trainNoise_News snr15_devNoise_News snr15_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr15_TalkShows.json" -w snr15_trainNoise_TalkShows snr15_devNoise_TalkShows snr15_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy_parted_snr15_Ambient_Music_top_charts.json" -w snr15_trainNoise_Ambient_Music_top_charts snr15_devNoise_Ambient_Music_top_charts snr15_testNoise_Ambient_Music_top_charts

    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr5_whiteNoise.json" -w snr5_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr5_Ads_2018_2020.json" -w snr5_trainNoise_Ads_2018_2020 snr5_devNoise_Ads_2018_2020 snr5_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr5_News.json" -w snr5_trainNoise_News snr5_devNoise_News snr5_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr5_TalkShows.json" -w snr5_trainNoise_TalkShows snr5_devNoise_TalkShows snr5_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr5_Ambient_Music_top_charts.json" -w snr5_trainNoise_Ambient_Music_top_charts snr5_devNoise_Ambient_Music_top_charts snr5_testNoise_Ambient_Music_top_charts && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr15_whiteNoise.json" -w snr15_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr15_Ads_2018_2020.json" -w snr15_trainNoise_Ads_2018_2020 snr15_devNoise_Ads_2018_2020 snr15_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr15_News.json" -w snr15_trainNoise_News snr15_devNoise_News snr15_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr15_TalkShows.json" -w snr15_trainNoise_TalkShows snr15_devNoise_TalkShows snr15_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy_snr15_Ambient_Music_top_charts.json" -w snr15_trainNoise_Ambient_Music_top_charts snr15_devNoise_Ambient_Music_top_charts snr15_testNoise_Ambient_Music_top_charts
    
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr5_whiteNoise.json" -w snr5_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr5_Ads_2018_2020.json" -w snr5_trainNoise_Ads_2018_2020 snr5_devNoise_Ads_2018_2020 snr5_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr5_News.json" -w snr5_trainNoise_News snr5_devNoise_News snr5_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr5_TalkShows.json" -w snr5_trainNoise_TalkShows snr5_devNoise_TalkShows snr5_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr5_Ambient_Music_top_charts.json" -w snr5_trainNoise_Ambient_Music_top_charts snr5_devNoise_Ambient_Music_top_charts snr5_testNoise_Ambient_Music_top_charts && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr15_whiteNoise.json" -w snr15_whiteNoise && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr15_Ads_2018_2020.json" -w snr15_trainNoise_Ads_2018_2020 snr15_devNoise_Ads_2018_2020 snr15_testNoise_Ads_2018_2020 && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr15_News.json" -w snr15_trainNoise_News snr15_devNoise_News snr15_testNoise_News && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr15_TalkShows.json" -w snr15_trainNoise_TalkShows snr15_devNoise_TalkShows snr15_testNoise_TalkShows && \
    python dataReducer.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy_snr15_Ambient_Music_top_charts.json" -w snr15_trainNoise_Ambient_Music_top_charts snr15_devNoise_Ambient_Music_top_charts snr15_testNoise_Ambient_Music_top_charts
    
    """
    reductionAmount = percentageReduction / 100
    dirname = os.path.dirname(outJson)
    if not os.path.exists(dirname): os.makedirs(dirname)

    samples = loadFromJson(jsonPath)
    trainSamples = {}
    devSamples   = {}
    testSamples  = {}
    for i, ID in enumerate(samples.keys()):
        sample = samples[ID]
        if sample["partition"] == "train": trainSamples[sample["ID"]] = sample.copy()
        if sample["partition"] == "dev":   devSamples[sample["ID"]]   = sample.copy()
        if sample["partition"] == "test":  testSamples[sample["ID"]]  = sample.copy()

    if reductionAmount > 0:
        print("Performing data reduction based on percentage")
        trainKeys = random.sample(trainSamples.keys(), int(reductionAmount*len(trainSamples.keys()))) 
        devKeys   = random.sample(devSamples.keys(), int(reductionAmount*len(devSamples.keys()))) 
        testKeys  = random.sample(testSamples.keys(), int(reductionAmount*len(testSamples.keys()))) 
        if keepTest: testKeys = testSamples.keys()
        newSamples = {}
        for keys in [trainKeys, devKeys, testKeys]:
            for ID in keys:
                sample = samples[ID]
                newSamples[ID] = sample.copy()
        print("Data reduction completed!")
    else:
        newSamples = samples.copy()

    for filterString in blackList:
        blackKeys = []
        for i, key in enumerate(newSamples.keys()):
            if filterString in key: blackKeys.append(key)
            printProgressBar(i+1, len(newSamples.keys()), prefix = 'removing black-listed IDs :', suffix = '', length = "fit")
        [newSamples.pop(key) for key in blackKeys]

    if len(whiteList) > 0:
        blackKeys = []
        for i, key in enumerate(newSamples.keys()):
            flag = True
            for whiteString in whiteList:
                if whiteString in key: flag = False
            if flag: blackKeys.append(key)
            printProgressBar(i+1, len(newSamples.keys()), prefix = 'Keeping only white-listed IDs :', suffix = '', length = "fit")
        [newSamples.pop(key) for key in blackKeys]

    with open(outJson, 'w') as jsonFile:
        json.dump(newSamples, jsonFile,  indent=4, ensure_ascii=False)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="input json file path")
    parser.add_argument('--outJson', '-o', help="output json file path with references to noisy json files")
    parser.add_argument('--percentageReduction', '-p', help="the amount of targeted reduction in percentage if that is the target (if not put 0 or ignore)", default=0.0)
    parser.add_argument('--keepTest', '-kt', help="keep test data in tact", default=False)
    parser.add_argument('--blackList', '-b', help="filter the IDs contating the strings given", nargs='+', default=[])
    parser.add_argument('--whiteList', '-w', help="filter the IDs not contating the strings given", nargs='+', default=[])

    args = parser.parse_args()
    Flag = False
    if args.json is None:    Flag=True
    if args.outJson is None: Flag=True
    if args.keepTest == "False": args.keepTest = False

    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.json, args.outJson, float(args.percentageReduction), args.keepTest, args.blackList, args.whiteList)

