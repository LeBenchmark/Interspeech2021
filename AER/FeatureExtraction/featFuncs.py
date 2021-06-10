from DatasetHandling.DataClasses import *
import json

def loadFromJson(jsonPath):
    jsonDict = {}
    with open(jsonPath, 'r') as jsonFile: 
        jsonDict = json.load(jsonFile)
    return jsonDict
