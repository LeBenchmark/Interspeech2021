# IMPORT MODELS
# IMPORT TORCH AND TORCH.NN
import json, os

class Experiment():
    def __init__(self, ID, loadPath=""):
        self.ID = ID
        self.description = "" # experiment description
        self.expPath = "" # path to save experiment related files such as results or logs.
        self.genre = "" # "classic" feat->annot, "e2e" wav->annot
        self.stage = "ready" # "ready", "trained", "tested"
        self.data = {
            "path": "", # path to data.json file
            "feature": "", # which feature to read (if needed, e.g. if not wav as input, can be left as "")
            "annotation": "" # path to data.json file (if needed, can be left as "")
            }
        self.testOn = "test"
        self.device = "cpu" # "cpu" or "cuda"
        self.batchSize = 1
        self.model = "Transformer"
        self.modelParams = "128-8-1"
        self.criterion = "CCC"
        self.optimizer = "Adam"
        self.learningRate = 0.001
        self.maxEpoch = 250
        self.tolerance = 15
        self.minForTolerance = 20
        self.inputDim = 40
        self.outputDim = 1
        self.limitTrainData = False
        self.limitDevData   = False
        self.evaluation = {"CCC":0.0, "MSE":0.0}
        if loadPath!="": self.loadFromJson(loadPath)

    # def getModel(self):
    #     models = {
    #         "Transformer": myTransformer(self.inputDim, lastOnly=True, outputSize=self.outputDim)
    #     }
    #     self.model = models[self.model]
    #     return self.model

    # def getOptimizer(self):
    #     optimizer = {
    #         "Adam": torch.optim.Adam(model.parameters(), lr=self.learningRate)
    #     }
    #     return optimizer[self.optimizer]

    # def getCriterion(self):
    #     pass

    def loadFromJson(self, jsonPath):
        with open(jsonPath, 'r') as jsonFile: 
            jsonDict = json.load(jsonFile)
            self.__dict__ = jsonDict[self.ID]

    def saveToJson(self, jsonPath, ignoreExisting=False):
        jsonDict = {}
        try:
            with open(jsonPath, 'r') as jsonFile: 
                jsonDict = json.load(jsonFile)
        except:
            if not os.path.exists(os.path.dirname(jsonPath)): os.makedirs(os.path.dirname(jsonPath))
        if ignoreExisting and (self.ID in jsonDict.keys()): 
            return
        jsonDict[self.ID] = self.__dict__
        with open(jsonPath, 'w') as jsonFile:
            json.dump(jsonDict, jsonFile,  indent=4, ensure_ascii=False)


# ex1 = Experiment("ex1")
# ex1.saveToJson("./experiments.json")
# ex2 = Experiment("ex2")
# ex2.stage = "trained"
# ex2.saveToJson("./experiments.json")
# ex2L = Experiment("ex2", loadPath="./experiments.json")
# print(ex2L.stage)

