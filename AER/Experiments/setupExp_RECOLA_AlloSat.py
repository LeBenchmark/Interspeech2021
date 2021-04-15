from Experiment import Experiment
import os

ExpPath  = "./Experiments/RECOLA_AlloSat/"
jsonPath = os.path.join(ExpPath, "experiments.json")
if os.path.exists(jsonPath): os.remove(jsonPath) # To remove from the json file, all the experiments before will be removed 
datasetsPath = "/mnt/HD-Storage/Datasets" # "/Users/sinaalisamir/Documents/Datasets"
device = "cuda"
ignoreExisting = True

datasets = ["RECOLA", "AlloSat"]
models = {
    "LinTanh": [""],
    "GRU": ["32-1", "64-1"],
    }
feats = ["MFB_standardized", "FlowBERT_2952h_base_cut30", "xlsr_53_56k_cut30"]

for model in models:
    for param in models[model]:
        for feat in feats:
            for dataset in datasets:
                annots = []
                if dataset=="RECOLA": 
                    annots = ["gs_arousal_0.01", "gs_valence_0.01"]
                    if feat in ["FlowBERT_2952h_base_cut30", "xlsr_53_56k_cut30"]:
                        annots = ["gs_arousal_0.02", "gs_valence_0.02"]
                if dataset=="AlloSat":
                    annots = ["labels_0.01"]
                    if feat in ["FlowBERT_2952h_base_cut30", "xlsr_53_56k_cut30"]:
                        annots = ["labels_0.02"]
                for annot in annots:
                    expID = dataset + "_" + annot + "_" + feat + "_" + model + "_" + param
                    myExp = Experiment(expID)
                    myExp.expPath = os.path.join(ExpPath, myExp.ID)
                    myExp.genre = "classic"
                    myExp.data = {
                        "path": os.path.join(datasetsPath, dataset, "data.json"), # path to data.json file
                        "feature": feat, # which feature to read (if needed, e.g. if not wav as input, can be left as "")
                        "annotation": annot # path to data.json file (if needed, can be left as "")
                        }
                    myExp.model = model
                    myExp.modelParams = param
                    myExp.device = device
                    myExp.criterion = "CCC"
                    myExp.optimizer = "Adam"
                    myExp.learningRate = 0.001
                    myExp.maxEpoch = 250
                    myExp.tolerance = 15
                    myExp.minForTolerance = 50
                    myExp.evaluation = {"CCC":0.0, "RMSE":0.0}
                    myExp.saveToJson(jsonPath, ignoreExisting)

