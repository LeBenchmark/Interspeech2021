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

def main(jsonPath, outJson, noiseFilesPaths, addWhite, SNRs, ignoreExisting):
    """
    Adding noise to a given dataset and making a different json file for it to reference it


    Example
    ----------
    python addNoise.py -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -o "/mnt/HD-Storage/Datasets/Recola_46/data_noisy.json" -aw True -n "/mnt/HD-Storage/Datasets/NoiseFiles/train" "/mnt/HD-Storage/Datasets/NoiseFiles/dev" "/mnt/HD-Storage/Datasets/NoiseFiles/test"
    python addNoise.py -j "/mnt/HD-Storage/Datasets/SEMAINE/data.json" -o "/mnt/HD-Storage/Datasets/SEMAINE/data_noisy.json" -aw True -n "/mnt/HD-Storage/Datasets/NoiseFiles/train" "/mnt/HD-Storage/Datasets/NoiseFiles/dev" "/mnt/HD-Storage/Datasets/NoiseFiles/test"
    
    python addNoise.py -j "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted.json" -o "/mnt/HD-Storage/Datasets/IEMOCAP/data_parted_noisy.json" -aw True -n "/mnt/HD-Storage/Datasets/NoiseFiles/train" "/mnt/HD-Storage/Datasets/NoiseFiles/dev" "/mnt/HD-Storage/Datasets/NoiseFiles/test"
    python addNoise.py -j "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted.json" -o "/mnt/HD-Storage/Datasets/MaSS_Fr/data_parted_noisy.json" -aw True -n "/mnt/HD-Storage/Datasets/NoiseFiles/train" "/mnt/HD-Storage/Datasets/NoiseFiles/dev" "/mnt/HD-Storage/Datasets/NoiseFiles/test"
    """
    datasetPath = os.path.split(jsonPath)[0]
    noisyFolder = "Wavs_Noisy"
    noisyWavsPath = os.path.join(datasetPath, noisyFolder)
    if not os.path.exists(noisyWavsPath): os.makedirs(noisyWavsPath)

    trainPath = os.path.join(noiseFilesPaths[0], "**", "*.wav")
    trainNoises = glob.glob(trainPath, recursive=True)
    devPath = os.path.join(noiseFilesPaths[1], "**", "*.wav")
    devNoises = glob.glob(devPath, recursive=True)
    testPath = os.path.join(noiseFilesPaths[2], "**", "*.wav")
    testNoises = glob.glob(testPath, recursive=True)

    samples = loadFromJson(jsonPath)
    newSamples = samples.copy()
    for i, ID in enumerate(samples.keys()):
        sample = samples[ID].copy()
        wavePath = sample["path"]
        wavFullPath = os.path.join(datasetPath, wavePath)

        sample["features"] = {} # to avoid reading the wrong feature extracted from clean speech

        wavsFolder = wavePath.split(os.sep)[0]
        splits = wavePath.split(os.sep)
        fileName = splits[-1].replace(".wav", "")

        ## MAKE NOISY FILES AND ADD TO SAMPLES, GIVE A NEW ID (which would be name of file)
        noiseFiles = trainNoises
        if sample["partition"] == "dev" : noiseFiles = devNoises
        if sample["partition"] == "test": noiseFiles = testNoises
        for snr in SNRs:
            for noiseFile in noiseFiles:
                outWavPath = noisyFolder
                for split in splits[1:-1]:
                    outWavPath = os.path.join(outWavPath, split) 
                outWavName = fileName +'_snr' + str(snr) + '_' + noiseFile.split(os.sep)[-1]
                outWavPath = os.path.join(outWavPath, outWavName)
                outWavFullPath = os.path.join(datasetPath, outWavPath)
                if not (ignoreExisting and os.path.exists(outWavFullPath)):
                    addNoiseFile(wavFullPath, noiseFile, outWavFullPath, snr=snr)
                ID = outWavName.replace(".wav", "")
                newSample = sample.copy()
                newSample["path"] = outWavPath
                newSample["ID"]   = ID
                newSamples[ID] = newSample

            if addWhite: 
                outWavPath = noisyFolder
                for split in splits[1:-1]:
                    outWavPath = os.path.join(outWavPath, split) 
                outWavName = fileName +'_snr' + str(snr) + '_whiteNoise.wav'
                outWavPath = os.path.join(outWavPath, outWavName)
                outWavFullPath = os.path.join(datasetPath, outWavPath)
                if not (ignoreExisting and os.path.exists(outWavFullPath)):
                    addWhiteNoise(wavFullPath, outWavFullPath, snr=snr)
                ID = outWavName.replace(".wav", "")
                newSample = sample.copy()
                newSample["path"] = outWavPath
                newSample["ID"]   = ID
                newSamples[ID] = newSample

        printProgressBar(i + 1, len(samples), prefix = 'Making wav files noisy:', suffix = 'Complete', length = "fit")


    with open(outJson, 'w') as jsonFile:
        json.dump(newSamples, jsonFile,  indent=4, ensure_ascii=False)


def whiteNoise(size):
    return np.random.normal(0, 1, size=size)

def noiseGain(snr, sGain=1):
    return 10 ** (math.log10(sGain) - snr/20)

def addWhiteNoise(filePath, outPath, snr=5):
    fileWav = wave.open(filePath, "r")
    fileVec = (np.frombuffer(fileWav.readframes(fileWav.getnframes()), dtype="int16")).astype(np.float64)
    noiseVec = whiteNoise(len(fileVec))
    addNoise(fileWav, fileVec, noiseVec, outPath, snr=snr)

def addNoiseFile(filePath, noisePath, outPath, snr=5):
    fileWav = wave.open(filePath, "r")
    noiseWav = wave.open(noisePath, "r")
    fileVec = (np.frombuffer(fileWav.readframes(fileWav.getnframes()), dtype="int16")).astype(np.float64)
    noiseVec = (np.frombuffer(noiseWav.readframes(noiseWav.getnframes()), dtype="int16")).astype(np.float64)
    start = random.randint(0, len(noiseVec)-len(fileVec))
    noiseVec = noiseVec[start: start + len(fileVec)]
    addNoise(fileWav, fileVec, noiseVec, outPath, snr=snr)

def addNoise(fileWav, fileVec, noiseVec, outPath, snr=5):
    fileGain = np.sqrt(np.mean(np.square(fileVec), axis=-1))
    noiseNormed = noiseVec / np.sqrt(np.mean(np.square(noiseVec), axis=-1))
    newNoise = noiseGain(snr, sGain=fileGain) * noiseNormed
    mixed = fileVec + newNoise
    outFile = wave.Wave_write(outPath)
    outFile.setparams(fileWav.getparams())
    outFile.writeframes(array.array('h', mixed.astype(np.int16)).tobytes())
    outFile.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="input json file path")
    parser.add_argument('--outJson', '-o', help="output json file path with references to noisy json files")
    parser.add_argument('--noiseFilesPaths', '-n', help="paths to folders containing noise wav files, put 3 folders to have training/dev/test, 1 folder path if you don't have partitioning!", nargs='+', default=[])
    parser.add_argument('--addWhite', '-aw', help="whether to add white noise or not", default=False)
    parser.add_argument('--SNRs', '-snr', help="Signal to Noise ratios for mixing the noise with speech files", nargs='+', default=[5, 15])
    parser.add_argument('--ignoreExisting', '-ie', help="ignore already existing noisy files paths", default=True)

    args = parser.parse_args()
    Flag = False
    if args.json is None:    Flag=True
    if args.outJson is None: Flag=True
    if args.addWhite == "False": args.addWhite = False
    if args.ignoreExisting == "False": args.ignoreExisting = False
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.json, args.outJson, args.noiseFilesPaths, args.addWhite, args.SNRs, args.ignoreExisting)

