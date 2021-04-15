import os, sys
import argparse
from funcs import get_files_in_path
from funcs import printProgressBar
from pydub import AudioSegment
import numpy as np
import pandas as pd
from featFuncs import loadFromJson
from DatasetHandling.DataClasses import Features, classToDic
import json

import torch
import torch.nn.functional as F
import fairseq
        
def main(featsFolder, jsonPath, modelPath, maxDur, normalised):
    """
    Adding wav2vec2 features to a given dataset and writting its reference to a json file


    Example
    ----------
    python wav2vec2.py -f "xlsr_53_56k_cut30" -d 29.99 -n True  -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/xlsr_53_56k.pt"
    python wav2vec2.py -f "FlowBERT_2952h_base_cut30_noNorm" -d 29.99 -n False -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/FlowBERT_2952h_base.pt"
    python wav2vec2.py -f "FlowBERT_2952h_base_cut30" -d 29.99 -n True -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/FlowBERT_2952h_base.pt"
    python wav2vec2.py -f "FlowBERT_2952h_large_cut30" -d 29.99 -n True -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/FlowBERT_2952h_large.pt"
    python wav2vec2.py -f "FlowBERT_2952h_large_noNorm_cut30" -d 29.99 -n False -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/FlowBERT_2952h_large.pt"
    
    python wav2vec2.py -f "xlsr_53_56k_cut30" -d 29.98 -n True  -j "/mnt/HD-Storage/Datasets/AlloSat/data.json" -m "/mnt/HD-Storage/Models/xlsr_53_56k.pt"
    python wav2vec2.py -f "FlowBERT_2952h_base_cut30" -d 29.99 -n False -j /home/getalp/alisamis/Datasets/AlloSat/data.json -m /home/getalp/nguyen35/flowbert_ssl_resources/wav2vec2.0_models/2952h_base/checkpoint_best.pt
    
    python wav2vec2.py -f "FlowBERT_2952h_base_cut30_noNorm" -d 29.99 -n False -j "/mnt/HD-Storage/Datasets/Recola_46_S/data.json" -m "/mnt/HD-Storage/Models/FlowBERT_2952h_base.pt"

    not working: python wav2vec2.py -f "wav2vec2-large-xlsr-53-french" -d 29.99 -n True -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/wav2vec2-large-xlsr-53-french.zip"

    python wav2vec2.py -f "mls_french_base_cut30" -d 29.98 -n False  -j "/home/getalp/alisamis/Datasets/AlloSat/data.json" -m "/home/getalp/nguyen35/flowbert_ssl_resources/wav2vec2.0_models/mls_french_base/checkpoint_best.pt"
    python wav2vec2.py -f "xlsr_53_56k_cut30" -d 29.98 -n True  -j "/home/getalp/alisamis/Datasets/AlloSat/data.json" -m "/home/getalp/alisamis/Models/xlsr_53_56k.pt"
    python wav2vec2.py -f "mls_french_base_cut30" -d 29.98 -n False  -j "/home/getalp/alisamis/Datasets/Recola_46/data.json" -m "/home/getalp/nguyen35/flowbert_ssl_resources/wav2vec2.0_models/mls_french_base/checkpoint_best.pt"
    python wav2vec2.py -f "mls_french_base_cut30" -d 29.98 -n False  -j "/mnt/HD-Storage/Datasets/Recola_46/data.json" -m "/mnt/HD-Storage/Models/mls_french_base/checkpoint_best.pt"
    
	python wav2vec2.py -f "libri960_big_cut30" -d 29.98 -n True  -j "/home/getalp/alisamis/Datasets/Recola_46/data.json" -m "/home/getalp/dinarelm/work/data/Exchance/wav2vec/models/libri960_big.pt"
	python wav2vec2.py -f "libri960_big_cut30" -d 29.98 -n True  -j "/home/getalp/alisamis/Datasets/AlloSat/data.json" -m "/home/getalp/dinarelm/work/data/Exchance/wav2vec/models/libri960_big.pt"
    """
    
    # cp = torch.load(modelPath, map_location=torch.device('cpu'))
    # model = Wav2VecModel.build_model(cp['args'], task=None)
    # model.load_state_dict(cp['model'])
    # model.eval()
    # cp = torch.load(modelPath, map_location=torch.device('cpu'))
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([modelPath])
    model = model[0]
    model.eval()

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
        dim = makeFeatsCsv(waveFullPath, featsFullPath, model, maxDur, normalised)
        if dim == 0: continue
        featsDict = getFeatsDict(dim, featsFolder, featsLocalPath)
        samples[ID]["features"][featsDict["ID"]] = featsDict
        # saveToJson(jsonPath, sample)
        printProgressBar(i + 1, len(samples), prefix = 'Adding wav2vec features:', suffix = 'Complete', length = "fit")
    with open(jsonPath, 'w') as jsonFile:
        json.dump(samples, jsonFile,  indent=4, ensure_ascii=False)

def getFeatsDict(dim, featsFolder, path):
    feats = Features()
    feats.setParams(featsFolder, "Acoustic", dim, path)
    return classToDic(feats)

def makeFeatsCsv(wavePath, outPath, model, maxDur, normalised):
    winlen=0.025; winstep=0.01
    audio_file = AudioSegment.from_wav(wavePath)
    feats, n_cutts = getFeatsFromAudio(audio_file, model, maxDur, normalised)
    dim = list(feats.shape)
    header = ['feat_'+str(i) for i in range(len(feats[0]))]
    df = pd.DataFrame(data=feats,columns=header)

    length = len(feats)
    # print(length, feats.shape)
    winstep = audio_file.duration_seconds/(length) # length -> length + n_cutts ? cuz each feat seem to lack one less number so each cut seem to reduce one from length 
    # print("winstep", winstep, audio_file.duration_seconds, length, feats.shape)
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
   
def getFeatsFromAudio(audio_file, model, maxDur, normalised): # ADD "DICE" LATER
    # maxDur = 30
    # audio_file = AudioSegment.from_wav(path)
    rate = audio_file.frame_rate
    duration = audio_file.duration_seconds
    audio_files = []
    if duration < maxDur:
        audio_files.append(audio_file)
    while duration >= maxDur:
        audio_files.append(audio_file[:maxDur*1000])
        audio_file = audio_file[maxDur*1000:]
        duration = audio_file.duration_seconds
        if duration < maxDur: audio_files.append(audio_file)
    feats = []
    n_cutts = len(audio_files)
    # audio_files.reverse()
    for audio_file in audio_files:
        sig = np.array(audio_file.get_array_of_samples())
        sig = sig / 32767.0 # scaling for 16-bit integer
        sigTensor = torch.from_numpy(sig).unsqueeze(0).float()
        if normalised:
            sigTensor = F.layer_norm(sigTensor, sigTensor.shape)
        c = model.extract_features(sigTensor, padding_mask=None, mask=False)[0]
        feat = c.squeeze(0).detach().numpy()
        feat = np.append(feat, feat[-1]).reshape(feat.shape[0]+1, feat.shape[1]) # repeating the last item to match size!
        if len(feats)==0: 
            feats = feat
        else:
            feats = np.concatenate((feats, feat), axis=0)
        # print(feats.shape)
    return feats, n_cutts

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--featsFolder', '-f', help="features folder name")
    parser.add_argument('--json', '-j', help="json file path")
    parser.add_argument('--modelPath', '-m', help="model path")
    parser.add_argument('--normalised', '-n', help="whether to put layer_norm for feeding inputs or not", default=False)
    parser.add_argument('--maxDur', '-d', help="maximum duration for cutting the file", default=15)
    args = parser.parse_args()
    Flag = False
    if args.featsFolder is None:  Flag=True
    if args.json is None:         Flag=True
    if args.modelPath is None:    Flag=True
    if args.normalised == "False": args.normalised = False
    if args.normalised == "True":  args.normalised = True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.featsFolder, args.json, args.modelPath, float(args.maxDur), args.normalised)
    # wavPath = "../../Data/WavsProcessed"
    # csvPath = "../../Data/Feats_MFB"
    # calc_mfbs(wavPath, csvPath, rate=16000)
