from DataClasses import *
import os, glob
import pandas as pd
import json
from funcs import printProgressBar

def main():
    Datasets_Path = "/mnt/HD-Storage/Datasets"
    main_path     = os.path.join(Datasets_Path, "Recola_46")
    wavs_path     = os.path.join(main_path, "Wavs")
    jsonPath      = os.path.join(main_path, "data.json")

    path = os.path.join(wavs_path, "**", "*.wav")
    filesPaths = glob.glob(path, recursive=True)

    allDics = {}

    for i, filePath in enumerate(filesPaths):
        baseName = os.path.basename(filePath)[:-4]
        fileDict = AudioSample()

        partition = ""
        if "train" in baseName: partition = "train"
        if "dev"   in baseName: partition = "dev"
        if "test"  in baseName: partition = "test"

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

def getAnnots(main_path, baseName):
    annots_path = "Annots"
    path = os.path.join(main_path, annots_path)
    IDs = ["gs_arousal_0.01", "gs_arousal_0.02", "gs_arousal_0.04", "gs_valence_0.01", "gs_valence_0.02", "gs_valence_0.04"]
    annots = {}
    for ID in IDs:
        annots[ID] = classToDic(getAnnotsBy(path, ID, baseName))
    return annots

def getAnnotsBy(path, folder, baseName):
    path = os.path.join(path, folder, baseName+".csv")
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)
    annots = Annotation()
    genre = "valence"
    if "arousal" in  folder: genre = "arousal"
    dimension = df.shape
    headers = ["GoldStandard"]
    annots.setParams(folder, genre, dimension, path, headers)
    annots.annotator_info.language = "French"
    return annots

if __name__== "__main__":
    main()


