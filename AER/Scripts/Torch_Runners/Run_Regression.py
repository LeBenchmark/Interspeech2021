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
from funcs import csvReader, printProgressBar, smooth


def main(jsonPath, testRun, setTarg, seed):
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
			trainModel(experiment, testRun, setTarg, seed)
			experiment.stage = "trained"
			experiment.saveToJson(jsonPath)
		if experiment.stage == "trained":
			writeModelOutput(experiment, testRun, seed)
			experiment.stage = "readyForTest"
			experiment.saveToJson(jsonPath)
		if experiment.stage == "readyForTest":
			testModel(experiment, testRun, setTarg, seed)
			experiment.stage = "tested"
			experiment.saveToJson(jsonPath)
		if experiment.stage == "tested":
			print("Model is trained and tested! results:", str(experiment.evaluation))


def getWrapper(experiment, getBest=False, seed=0):
	wrapper = ModelWrapper(savePath=experiment.expPath, device=experiment.device, seed=seed, printLvl=10)
	wrapper.setModel(experiment.model, experiment.inputDim, experiment.outputDim, experiment.modelParams)
	wrapper.setOptimizer(experiment.optimizer, learningRate=experiment.learningRate)
	wrapper.setCriterion(experiment.criterion)
	wrapper.loadCheckpoint()
	if getBest: wrapper.loadBestModel()
	return wrapper


def trainModel(experiment, testRun, setTarg, seed=0):
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
	if not "classification" in experiment.genre:
		experiment.outputDim = tar.shape[1]
	# print("experiment.outputDim", tar.shape)
	wrapper = getWrapper(experiment, seed=seed)
	wrapper.trainModel(datasetTrain, datasetDev, batchSize=experiment.batchSize, maxEpoch=experiment.maxEpoch, 
					   loadBefore=True, tolerance = experiment.tolerance, 
					   minForTolerance=experiment.minForTolerance)
	wrapper.saveLogToCSV()
	

def writeModelOutput(experiment, testRun, seed=0):
	# print("Writing outputs ...")
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetFeatOnly("test", experiment.data["feature"])
	if testRun: dataset = keepOne(dataset)
	dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
	wrapper = getWrapper(experiment, seed=seed, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "outputs")
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


def writeModelOutput(experiment, testRun, seed=0):
	# print("Writing outputs ...")
	if "classification" in experiment.genre:
		writeOutForClassification(experiment, testRun, seed=seed)
	else:
		writeOutForRegression(experiment, testRun, seed=seed)
	

def writeOutForRegression(experiment, testRun, seed=0):
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetFeatOnly("test", experiment.data["feature"])
	if testRun: dataset = keepOne(dataset)
	dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
	wrapper = getWrapper(experiment, seed=seed, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "outputs")
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

def writeOutForClassification(experiment, testRun, seed=0):
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetFeatOnly("test", experiment.data["feature"])
	if testRun: dataset = keepOne(dataset)
	dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
	wrapper = getWrapper(experiment, seed=seed, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "outputs")
	if not os.path.exists(modelOutPath): os.makedirs(modelOutPath)
	headers = []
	outputs = []
	for idx, (ID, feat) in enumerate(dataloader):
		printProgressBar(idx+1, len(dataloader), prefix = 'Writing outputs:', suffix = '', length = "fit")
		output = wrapper.forwardModel(feat)
		output = output.detach().cpu().numpy()
		# print(ID, feat.shape, output.shape)
		outputs = output if len(outputs)==0 else np.concatenate((outputs,output))
		headers.append(ID[0])
		# print(len(headers), outputs.shape)
	savePath = os.path.join(modelOutPath, "outputs.csv")
	df = pd.DataFrame(np.transpose(outputs), columns = headers)
	df.to_csv(savePath, index=False)


def testModel(experiment, testRun, setTarg, seed=0):
	if "classification" in experiment.genre:
		return testClassification(experiment, testRun, setTarg, seed=seed)
	else:
		return testRegression(experiment, testRun, setTarg, seed=seed)


######### NOT CORRECT IN CASE OF -> if setTarg == "MeanStd" ###############
######### Bad coding, need fix, the key can be put inside, no need to run the loop multiple times! ###############
######### put concated flag outside to controle, change code to class!
def testRegression(experiment, testRun, setTarg, seed=0, concated=True):
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetAnnotOnly("test", experiment.data["annotation"])
	if setTarg == "MeanStd": dataset.setTargetMeanStd()
	idx, tar = dataset[0]
	# experiment.inputDim = inp.shape[1]
	# if not "classification" in experiment.genre:
	experiment.outputDim = tar.shape[1]

	firstID1 = list(dataset.dataPart.keys())[0]
	firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
	headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
	if setTarg == "MeanStd": headers = ["mean", "std"]
	# print(headers)
	wrapper = getWrapper(experiment, seed=seed, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "outputs")
	if testRun: dataset = keepOne(dataset)
	IDs = dataset.dataPart.keys()
	for key in experiment.evaluation.keys():
		metrics = {}
		allTars = []
		allOuts = []
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
				if concated: allTars+=list(target);  allOuts+=list(output)
				result = getMetric(target, output, metric=key)
				# if result > bestresult: bestresult=result; bestID = ID
				# print(ID, result, len(output))
				results[dim].append(result)
			printProgressBar(idx+1, len(IDs), prefix = 'Testing model with '+ key +':', suffix = '', length = "fit")
		for dim in range(targets.shape[1]): 
			if concated: 
				metrics[headers[dim]] = getMetric(np.array(allTars), np.array(allOuts), metric=key)
			else:
				metrics[headers[dim]]['mean'] = np.mean(np.array(results[dim]))
				metrics[headers[dim]]['std'] = np.std(np.array(results[dim]))
		experiment.evaluation[key] = metrics
	return experiment


def testClassification(experiment, testRun, setTarg, seed=0):
	dataset = DataReader(experiment.data["path"])
	dataset.setDatasetClassic("test", experiment.data["feature"], experiment.data["annotation"])
	inp, tar = dataset[0]
	experiment.inputDim = inp.shape[1]
	firstID1 = list(dataset.dataPart.keys())[0]
	firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
	headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
	# print(headers)
	wrapper = getWrapper(experiment, seed=seed, getBest=True)
	modelOutPath = os.path.join(wrapper.savePath, "outputs")
	savePath = os.path.join(modelOutPath, "outputs.csv")
	outputsCSV = pd.read_csv(savePath)
	if testRun: dataset = keepOne(dataset)
	IDs = dataset.dataPart.keys()
	AllOuts = []
	AllTars = []
	for idx, ID in enumerate(IDs):
		outputs = outputsCSV[ID].to_numpy()
		targets = dataset.targetReader(ID)
		AllOuts.append(np.argmax(outputs))
		AllTars.append(targets[0,0])
		# print(np.argmax(outputs), targets[0,0])
		printProgressBar(idx+1, len(IDs), prefix = 'Testing model :', suffix = '', length = "fit")
		# if idx > 50: break
	target = np.array(AllTars)
	output = np.array(AllOuts)
	metrics = {}
	for key in experiment.evaluation.keys():
		metrics[key] = getMetric(target, output, metric=key)
	experiment.evaluation = metrics
	confMat = confMatrix(target, output, numTars=experiment.outputDim)
	# print(confMatrix(target, output, numTars=experiment.outputDim))
	savePath = os.path.join(wrapper.savePath, "confMat.csv")
	np.savetxt(savePath, confMat, delimiter=",")
	return experiment


def getMetric(target, output, metric="CCC"):
	if metric == "CCC":
		result = CCC(target, output)
	if metric == "MSE":
		result = MSE(target, output)
	if metric == "RMSE":
		result = RMSE(target, output)
	if metric == "Accuracy":
		result = Accuracy(target, output)
	if metric == "UAR":
		from sklearn.metrics import recall_score
		result = recall_score(target, output, average='macro')
	if metric == "AUC":
		from sklearn.metrics import roc_auc_score
		result = roc_auc_score(target, output)
	if "AUC-" in metric:
		from sklearn.metrics import roc_auc_score
		result = roc_auc_score(target, smooth(output, win=int(metric.split('-')[1])))
	if "SmoothAcc-" in metric:
		result = Accuracy(target, smooth(output, win=int(metric.split('-')[1])))
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
    parser.add_argument('--seed', '-s', help="seed used for random number generation by torch", default=0)
    args = parser.parse_args()
    Flag = False
    if args.json is None: Flag=True
    if Flag:
        parser.print_help()
    else:
        main(args.json, args.testRun, args.setTarg, int(args.seed))
