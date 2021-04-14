from Experiment import Experiment
import os

ExpPath  = "../../Experiments/Recola_46_8/"
jsonPath = os.path.join(ExpPath, "experiments.json")
if os.path.exists(jsonPath): os.remove(jsonPath) # To remove from the json file, all the experiments before will be removed 
datasetsPath = "/mnt/HD-Storage/Datasets" # "/Users/sinaalisamir/Documents/Datasets"
device = "cuda"
ignoreExisting = True

datasets = ["Recola_46"]#, "Recola_46", "AlloSat", "SEWA_DE", "SEWA_HU", "SEWA_CN"]
models = {
    "LinTanh": [""],
    "GRU": ["32-1", "64-1", "128-1"],
    # "Transformer": ["32-1-1", "64-1-1", "128-1-1", "128-4-1", "128-4-2"]
    }
feats = ["FlowBERT_2952h_base_cut30_noNorm"]#"MFB_standardized", "FlowBERT_2952h_base_cut30", "xlsr_53_56k_cut30"

for model in models:
    for param in models[model]:
        for feat in feats:
            for dataset in datasets:
                annots = []
                if dataset=="Recola_46": 
                    annots = ["gs_arousal_0.01", "gs_valence_0.01"]
                    if feat in ["FlowBERT_2952h_base_cut30", "FlowBERT_2952h_large_cut30", "FlowBERT_2952h_base_cut30_noNorm", "FlowBERT_2952h_large_noNorm_cut30", "xlsr_53_56k_cut30"]:
                        annots = ["gs_arousal_0.02", "gs_valence_0.02"]
                if dataset=="AlloSat":
                    annots = ["labels_0.01"]
                    if feat in ["FlowBERT_2952h_base_cut30", "FlowBERT_2952h_large_cut30", "FlowBERT_2952h_base_cut30_noNorm", "FlowBERT_2952h_large_noNorm_cut30", "xlsr_53_56k_cut30"]:
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


# myExp = Experiment("RECOLA46_gs_arousal")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> arousal for RECOLA46 by Transformers"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "Recola_46", "data.json"), # path to data.json file
#     "feature": "MFB_standardized", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "Transformer"
# myExp.modelParams = "128-8-1"
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_FlowBERT_2952h_base_cut30_GRU_128")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "FlowBERT -> arousal for RECOLA46 by GRU"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "Recola_46", "data.json"), # path to data.json file
#     "feature": "FlowBERT_2952h_base_cut30", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.02" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "GRU"
# myExp.modelParams = "128"
# myExp.inputDim = 768
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_FlowBERT_2952h_base_cut30_LinTanh")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "FlowBERT -> arousal for RECOLA46 by LinTanh"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "Recola_46", "data.json"), # path to data.json file
#     "feature": "FlowBERT_2952h_base_cut30", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.02" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "myLinTanh"
# myExp.modelParams = ""
# myExp.inputDim = 768
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_FlowBERT_2952h_base_cut30_Transformer_128-8-1")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "FlowBERT -> arousal for RECOLA46 by Transformers"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "Recola_46", "data.json"), # path to data.json file
#     "feature": "FlowBERT_2952h_base_cut30", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.02" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "Transformer"
# myExp.modelParams = "128-8-1"
# myExp.inputDim = 768
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_valence")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> valence for RECOLA46 by Transformers"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "RECOLA_46_P", "data.json"), # path to data.json file
#     "feature": "MFB_S", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_valence_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "Transformer"
# myExp.modelParams = "128-8-1"
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_2")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> arousal for RECOLA46 by Transformers"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "RECOLA_46_P", "data.json"), # path to data.json file
#     "feature": "MFB_S", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "Transformer"
# myExp.modelParams = "128-8-2"
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_3")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> arousal for RECOLA46 by GRU"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "RECOLA_46_P", "data.json"), # path to data.json file
#     "feature": "MFB_S", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "GRU"
# myExp.modelParams = "128"
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_4")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> arousal for RECOLA46 by Sinc Transformers"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "RECOLA_46_P", "data.json"), # path to data.json file
#     "feature": "MFB_S", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "SincTransformer"
# myExp.modelParams = "128-8-1-100-10"
# myExp.device = device
# myExp.saveToJson(jsonPath)

# myExp = Experiment("RECOLA46_gs_arousal_5")
# myExp.expPath = "../../Experiments/" + myExp.ID
# myExp.description = "MFB -> arousal for RECOLA46 by Sinc Linear Tanh Sinc"
# myExp.genre = "classic"
# myExp.data = {
#     "path": os.path.join(datasetsPath, "RECOLA_46_P", "data.json"), # path to data.json file
#     "feature": "MFB_S", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
#     "annotation": "gs_arousal_0.01" # path to data.json file (if needed, can be left as "")
#     }
# myExp.model = "LinTanhSinc"
# myExp.modelParams = "64-100-10"
# myExp.device = device
# myExp.saveToJson(jsonPath)

