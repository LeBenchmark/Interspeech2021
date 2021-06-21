import os, sys
import argparse
from Utils.Funcs import get_files_in_path, printProgressBar, loadFromJson
from pydub import AudioSegment
from python_speech_features import logfbank
import numpy as np
import pandas as pd
from DataProcess.DataClasses import Features, classToDic
import json

def main(featsFolder, jsonPath):
    """
    Adding mel-frequency filterbank features to a given dataset and writting its reference to a json file


    Example
    ----------
    python MelFilterBank.py -f "MFB" -j "/mnt/HD-Storage/Datasets/Recola_46/data.json"
    python MelFilterBank.py -f "MFB" -j "/mnt/HD-Storage/Datasets/AlloSat/data.json"
    python MelFilterBank.py -f "MFB" -j "/mnt/HD-Storage/Datasets/Recola_46_S/data.json"
    
    python MelFilterBank.py -f "MFB" -j "/mnt/HD-Storage/Datasets/ESTER1/data.json"
    python MelFilterBank.py -f "MFB" -j "/home/getalp/alisamis/Datasets/ESTER1/data.json"

    python MelFilterBank.py -f "MFB" -j "/mnt/HD-Storage/Datasets/RAVDESS/data.json"
    """
    samples = loadFromJson(jsonPath)
    for i, ID in enumerate(samples.keys()):
        sample = samples[ID]
        wavePath = sample["path"]
        wavsFolder = wavePath.split(os.sep)[0]
        waveFullPath = os.path.join(os.path.split(jsonPath)[0], wavePath)
        featsLocalPath = wavePath.replace(wavsFolder, featsFolder).replace(".wav", ".csv")
        featsLocalPath = os.path.join("Feats", featsLocalPath)
        featsFullPath  = os.path.join(os.path.split(jsonPath)[0], featsLocalPath)
        
        # print(featsLocalPath, featsFullPath)
        dim = makeFeatsCsv(waveFullPath, featsFullPath)
        if dim == 0: continue
        featsDict = getFeatsDict(dim, featsFolder, featsLocalPath)
        samples[ID]["features"][featsDict["ID"]] = featsDict
        # saveToJson(jsonPath, sample)
        printProgressBar(i + 1, len(samples), prefix = 'Adding mel-frequency filterbank features:', suffix = 'Complete', length = "fit")
    with open(jsonPath, 'w') as jsonFile:
        json.dump(samples, jsonFile,  indent=4, ensure_ascii=False)

def getFeatsDict(dim, featsFolder, path):
    feats = Features()
    feats.setParams(featsFolder, "Acoustic", dim, path)
    return classToDic(feats)

def makeFeatsCsv(wavePath, outPath):
    winlen=0.025; winstep=0.01
    fbank_feat = getFeatsFromWav(wavePath, winlen=winlen, winstep=winstep)
    dim = list(fbank_feat.shape)
    header = ['feat_'+str(i) for i in range(len(fbank_feat[0]))]
    df = pd.DataFrame(data=fbank_feat,columns=header)

    length = len(fbank_feat)
    timesStart = []
    timesEnd = []
    time = 0
    for _ in range(length):
        timesStart.append(time)
        timesEnd.append(time + winlen)
        time += winstep
    df.insert(0, 'frameTimeEnd', timesEnd)
    df.insert(0, 'frameTimeStart', timesStart)
    
    dirname = os.path.dirname(outPath)
    if not os.path.exists(dirname): os.makedirs(dirname)
    with open(outPath, 'w', encoding='utf-8') as f:
        df.to_csv(f, index=False)
    return dim
   
def getFeatsFromWav(path, winlen=0.025, winstep=0.01):
    audio_file = AudioSegment.from_wav(path)
    sig = np.array(audio_file.get_array_of_samples())
    sig = sig / 32767.0 # scaling for 16-bit integer
    rate = audio_file.frame_rate
    fbank_feat = logfbank(sig, rate, winlen=winlen, winstep=winstep, nfilt=40, nfft=2028, lowfreq=0, highfreq=None, preemph=0.97) #, winfunc=np.hanning
    return fbank_feat

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--featsFolder', '-f', help="features folder name")
    parser.add_argument('--json', '-j', help="json file path")
    args = parser.parse_args()
    Flag = False
    if args.featsFolder is None:  Flag=True
    if args.json is None:         Flag=True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.featsFolder, args.json)
    # wavPath = "../../Data/WavsProcessed"
    # csvPath = "../../Data/Feats_MFB"
    # calc_mfbs(wavPath, csvPath, rate=16000)
