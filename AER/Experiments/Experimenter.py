from Experiment import Experiment
import json
import argparse
from modelWrapper import ModelWrapper
from DataReader import DataReader
import pandas as pd
import numpy as np
# import torch
from torch.utils.data import DataLoader
import os, sys
from Metrics import *
from funcs import csvReader, printProgressBar


def main(jsonPath, testRun, setTarg):
	"""
	example:
		python Experimenter.py -j "../../Experiments/experiments.json"
	"""
	# jsonPath = "./experiments.json"
	expIDs = getExpIDs(jsonPath)
	for expID in expIDs:
		experiment = Experiment(expID, loadPath=jsonPath)
		print("--- Experiment with ID:", experiment.ID, "---")
		if experiment.stage == "ready":
			trainModel(experiment, testRun, setTarg)
			experiment.stage = "trained"
			experiment.saveToJson(jsonPath)
		if experiment.stage == "trained":
			writeModelOutput(experiment, testRun)
			testModel(experiment, testRun, setTarg)
			experiment.stage = "tested"
			experiment.saveToJson(jsonPath)
		if experiment.stage == "tested":
			print("Model is trained and tested! results:", str(experiment.evaluation))


def getWrapper(experiment, getBest=False):
	wrapper = ModelWrapper(savePath=experiment.expPath, device=experiment.device, printLvl=10)
	wrapper.setModel(experiment.model, experiment.inputDim, experiment.outputDim, experiment.modelParams)
	wrapper.setOptimizer(experiment.optimizer, learningRate=experiment.learningRate)
	wrapper.setCriterion(experiment.criterion)
	wrapper.loadCheckpoint()
	if getBest: wrapper.loadBestModel()
	return wrapper


def trainModel(experiment, testRun, setTarg):
	print("Training model ...")

	datasetTrain = DataReader(experiment.data["path"])
	datasetTrain.setDatasetClassic("train", experiment.data["feature"], experiment.data["annotation"])
	if setTarg == "MeanStd": datasetTrain.setTargetMeanStd()
	if testRun: datasetTrain = keepOne(datasetTrain)
	datasetDev = DataReader(experiment.data["path"])
	datasetDev.setDatasetClassic("dev", experiment.data["feature"], experiment.data["annotation"])
	if setTarg == "MeanStd": datasetDev.setTargetMeanStd()
	if testRun: datasetDev = keepOne(datasetDev)
	if testRun: experiment.maxEpoch = 1
	inp, tar = datasetDev[0]
	experiment.inputDim = inp.shape[1]
	experiment.outputDim = tar.shape[1]
	# print("experiment.outputDim", tar.shape)
	wrapper = getWrapper(experiment)
	wrapper.trainModel(datasetTrain, datasetDev, batchSize=experiment.batchSize, maxEpoch=experiment.maxEpoch, 
					   loadBefore=True, tolerance = experiment.tolerance, 
					   minForTolerance=experiment.minForTolerance)
	wrapper.saveLogToCSV()
	

def writeModelOutput(experiment, testRun):
	# print("Writing outputs ...")
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetFeatOnly("test", experiment.data["feature"])
	if testRun: dataset = keepOne(dataset)
	dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
	wrapper = getWrapper(experiment, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "ouputs")
	if not os.path.exists(modelOutPath): os.makedirs(modelOutPath)
	for idx, (ID, feat) in enumerate(dataloader):
		output = wrapper.forwardModel(feat)
		output = output.detach().cpu().numpy()
		# print(ID, feat.shape, output.shape)
		savePath = os.path.join(modelOutPath, ID[0]+".csv")
		headers = ["output_"+str(i) for i in range(output.shape[2])]
		df = pd.DataFrame(output[0], columns = headers)
		df.to_csv(savePath, index=False)
		printProgressBar(idx+1, len(dataloader), prefix = 'Writing outputs:', suffix = '', length = "fit")


def testModel(experiment, testRun, setTarg):
	# print("Testing model ...")
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetClassic("test", experiment.data["feature"], experiment.data["annotation"])
	if setTarg == "MeanStd": dataset.setTargetMeanStd()
	inp, tar = dataset[0]
	experiment.inputDim = inp.shape[1]
	experiment.outputDim = tar.shape[1]

	firstID1 = list(dataset.dataPart.keys())[0]
	firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
	headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
	if setTarg == "MeanStd": headers = ["mean", "std"]
	# print(headers)
	wrapper = getWrapper(experiment, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "ouputs")
	if testRun: dataset = keepOne(dataset)
	IDs = dataset.dataPart.keys()
	for key in experiment.evaluation.keys():
		metrics = {}
		for idx, ID in enumerate(IDs):
			savePath = os.path.join(modelOutPath, ID+".csv")
			outputs = pd.read_csv(savePath).to_numpy()
			targets = dataset.targetReader(ID)
			# print(targets.shape, outputs.shape)
			if idx == 0:
				results = [[] for _ in range(targets.shape[1])]
				# bestresult = 0; bestID = "0"
				for dim in range(targets.shape[1]): metrics[headers[dim]] = {} 
			for dim in range(targets.shape[1]):
				output = outputs[:, dim]
				target = targets[:, dim]
				while target.shape[0] > output.shape[0]: output = np.append(output, outputs[-1])
				while target.shape[0] < output.shape[0]: output = outputs[:target.shape[0]].reshape(target.shape[0])
				result = getMetric(target, output, metric=key)
				# if result > bestresult: bestresult=result; bestID = ID
				# print(ID, result, len(output))
				results[dim].append(result)
			printProgressBar(idx+1, len(IDs), prefix = 'Testing model with '+ key +':', suffix = '', length = "fit")
		for dim in range(targets.shape[1]): 
			metrics[headers[dim]]['mean'] = np.mean(np.array(results[dim]))
			metrics[headers[dim]]['std'] = np.std(np.array(results[dim]))
		experiment.evaluation[key] = metrics
	return experiment
	# a = csvReader(savePath, headers)


def getMetric(target, output, metric="CCC"):
	if metric == "CCC":
		result = CCC(target, output)
	if metric == "MSE":
		result = MSE(target, output)
	if metric == "RMSE":
		result = RMSE(target, output)
	return result


def getExpIDs(jsonPath):
	with open(jsonPath, 'r') as jsonFile: 
	    jsonDict = json.load(jsonFile)
	    expIDs = jsonDict.keys()
	return expIDs


def keepOne(dataset):
	blackKeys = []
	for i, key in enumerate(dataset.dataPart.keys()):
		if i==0: continue
		blackKeys.append(key)
	[dataset.dataPart.pop(key) for key in blackKeys]
	return dataset


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="path to experiments.json file")
    parser.add_argument('--testRun', '-t', help="testing by running only on one data", default=False)
    parser.add_argument('--setTarg', '-g', help="set target to another thing, \"original\" keeps target as is., \"MeanStd\" changes it to its mean and std.", default="original")
    args = parser.parse_args()
    Flag = False
    if args.json is None: Flag=True
    if Flag:
        parser.print_help()
    else:
        main(args.json, args.testRun, args.setTarg)
