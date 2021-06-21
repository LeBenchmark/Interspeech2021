import os, sys, argparse, json, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from DataProcess.Preprocess import main as Preprocess
from DataProcess.makeJson import main as makeJson
from DataProcess.addAnnots import main as addAnnots
from Experiments.Experiment import Experiment
from Torch_Runners.Run_main import Runner

def main(wavsFolder, annotsFolder, outFolder):
    """
    example:
        python RECOLA_2016.py -w "/mnt/HD-Storage/Datasets/RECOLA_2016/recordings_audio" -a "/mnt/HD-Storage/Datasets/RECOLA_2016/ratings_gold_standard" -o "/mnt/HD-Storage/Datasets/RECOLA_2016"
    """
    
    ## Preprocess wav files
    print("Preprocess wav files...")
    newFolder = os.path.join(outFolder, "Wavs")
    Preprocess(wavsFolder, newFolder, False, "wav")

    ## Create json file
    makeJson(outFolder, ["train_*.wav", "dev_*.wav", "test_*.wav"])

    ## Create labels
    mainFolder = os.path.split(annotsFolder)[0]
    arffs2csvs(annotsFolder, "Annots", mainFolder, outFolder)

    ## add Annots
    dataJsonPath = os.path.join(outFolder, "data.json")
    genres = ["arousal", "valence"]
    annotsList = ["arousal", "valence"]
    headers = ["GoldStandard"]
    addAnnots(annotsList, headers, genres, dataJsonPath)

    ## Set the experiments
    ExpPath     = os.path.join(outFolder, "Results")
    expJsonPath = os.path.join(ExpPath, "experiments.json")
    testOn = "dev" # the results will be reported for the dev set!
    setExperiments(outFolder, dataJsonPath, ExpPath, expJsonPath, testOn)

    ## Run the experiments
    seed = 0
    testRun = False
    testConcated = True
    onlineFeat = True
    resampleTarget = True
    runner = Runner(expJsonPath, seed, testRun, testConcated, onlineFeat, resampleTarget)
    runner.main()


def setExperiments(outFolder, dataJsonPath, ExpPath, expJsonPath, testOn):
    if os.path.exists(expJsonPath): os.remove(expJsonPath) # To remove from the json file, all the experiments before will be removed 
    device = "cuda"
    ignoreExisting = True

    models = {
        "LinTanh": [""],
        "GRU": ["32-1", "64-1"]
        }
    feats = ["MFB_standardized", "wav2vec2-FlowBERT_2952h_base"] #"MFB_standardized", "wav2vec2-xlsr_53", "wav2vec2-FlowBERT_2952h_base"
    featModelPaths = {"MFB_standardized": "",
                      "wav2vec2-FlowBERT_2952h_base": "/mnt/HD-Storage/Models/FlowBERT_2952h_base.pt",
                     }
    annots = ["arousal", "valence"]

    for model in models:
        for param in models[model]:
            for feat in feats:
                featModelPath = ""
                featModelNorm = True
                featModelMaxDur = 15.98
                if feat == "wav2vec2-FlowBERT_2952h_base": 
                    featModelNorm = False
                for annot in annots:
                    expID = annot + "_" + feat + "_" + model + "_" + param
                    myExp = Experiment(expID)
                    myExp.expPath = os.path.join(ExpPath, myExp.ID)
                    myExp.genre = "classic"
                    myExp.data = {
                        "path": dataJsonPath, # path to data.json file
                        "devPath": dataJsonPath,
                        "testPaths": [dataJsonPath],
                        "annotation": annot, # path to data.json file (if needed, can be left as "")
                        "feature": feat, # which feature to read (if needed, e.g. if not wav as input, can be left as "")
                        "featModelPath": featModelPaths[feat],
                        "featModelNorm": featModelNorm,
                        "featModelMaxDur": featModelMaxDur,
                        }
                    myExp.testOn = testOn
                    myExp.model = model
                    myExp.modelParams = param
                    myExp.device = device
                    myExp.criterion = "CCC"
                    myExp.optimizer = "Adam"
                    myExp.learningRate = 0.001
                    myExp.maxEpoch = 250
                    myExp.tolerance = 15
                    myExp.minForTolerance = 50
                    myExp.metrics = ["CCC", "RMSE"]
                    myExp.evaluation = {}
                    myExp.saveToJson(expJsonPath, ignoreExisting)

def arffs2csvs(arffFolder, csvFolder, mainFolder, outFolder):
    filePath = os.path.join(arffFolder, "**", "*.arff")
    filesPaths = glob.glob(filePath, recursive=True)
    outFiles = [filePath.replace(mainFolder, outFolder).replace(os.path.split(arffFolder)[1], csvFolder).replace(".arff", ".csv") for filePath in filesPaths]
    for i, filePath in enumerate(filesPaths):
        outPath = outFiles[i]
        directory = os.path.dirname(outPath)
        if not os.path.exists(directory): os.makedirs(directory)
        arff2csv(filePath, csv_path=outPath)

def arff2csv(arff_path, csv_path=None, _encoding='utf8'):
    with open(arff_path, 'r', encoding=_encoding) as fr:
        attributes = []
        if csv_path is None:
            csv_path = arff_path[:-4] + 'csv'  # *.arff -> *.csv
        write_sw = False
        with open(csv_path, 'w', encoding=_encoding) as fw:
            for line in fr.readlines():
                if write_sw:
                    if line == "": print("emp")
                    if line != "\n": fw.write(line)
                elif '@data' in line:
                    fw.write(','.join(attributes) + '\n')
                    write_sw = True
                elif '@attribute' in line:
                    attributes.append(line.split()[1])  # @attribute attribute_tag numeric
    print("Convert {} to {}.".format(arff_path, csv_path))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotsFolder', '-a', help="path to folder contatining annotations files", default="")
    parser.add_argument('--wavsFolder', '-w', help="path to folder containing wav files", default="")
    parser.add_argument('--outFolder', '-o', help="path to folder containing output files", default="")
    
    args = parser.parse_args()
    Flag = False
    if args.annotsFolder == "": Flag = True
    if args.wavsFolder == "":  Flag = True
    if args.outFolder == "":   Flag = True
    if Flag:
        parser.print_help()
    else:
        main(args.wavsFolder, args.annotsFolder, args.outFolder)

