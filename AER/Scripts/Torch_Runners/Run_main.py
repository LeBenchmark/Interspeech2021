import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from Experiments.Experiment import Experiment
import json
import argparse
from Torch_Runners.modelWrapper import ModelWrapper
from DataProcess.DataReader import DataReader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os, sys
from Utils.Metrics import getMetric
from Utils.Funcs import csvReader, printProgressBar, smooth

class Runner(object):
	"""
	This class is used to run experiments! Based on indicated parameters, this main class routes to scripts for training and testing models.
	Example:
		python Run_main.py -j "../../Experiments/Recola_46_11/experiments.json"
		python Run_main.py -j "../../Experiments/Recola_46_12/experiments.json" -rt True -o True -t True
		python Run_main.py -j "../../Experiments/Semaine_VAD1/experiments.json" -rt True -o True -c True
		python Run_main.py -j "../../Experiments/Semaine_VAD2/experiments.json" -rt True -o True -c True
	code fixes:
		pass experiment to dataset to avoid all these passing parameters?
		put different parts like classification and regression codes into different files!
	"""
	def __init__(self, jsonPath, seed=0, testRun=False, testConcated=True, onlineFeat=False, resampleTarget=False):
		super(Runner, self).__init__()
		self.jsonPath = jsonPath
		self.seed = seed
		self.testRun = testRun
		self.testConcated = testConcated
		self.onlineFeat = onlineFeat
		self.resampleTarget = resampleTarget
		self.cuda = False


	def main(self):
		expIDs = self.getExpIDs()
		for expID in expIDs:
			self.experiment = Experiment(expID, loadPath=self.jsonPath)
			if "cuda" in self.experiment.device: self.cuda = True
			self.limitTrainData = self.experiment.limitTrainData
			self.limitDevData   = self.experiment.limitDevData 
			print("--- Experiment with ID:", self.experiment.ID, "---")
			if self.experiment.stage == "ready":
				self.trainModel()
				self.experiment.stage = "trained"
				self.experiment.saveToJson(self.jsonPath)
			if self.experiment.stage == "trained":
				print("Model is trained!")
				for path in self.experiment.data["testPaths"]:
					self.currentDataPath = path
					self.writeModelOutput()
				self.experiment.stage = "readyForTest"
				self.experiment.saveToJson(self.jsonPath)
			if self.experiment.stage == "readyForTest":
				print("Model outputs are extracted and ready for test!")
				evaluations = {}
				for path in self.experiment.data["testPaths"]:
					self.currentDataPath = path
					evaluations[path] = self.testModel()
				self.experiment.evaluation = evaluations
				self.experiment.stage = "tested"
				self.experiment.saveToJson(self.jsonPath)
			if self.experiment.stage == "tested":
				print("Model is tested! results:", str(self.experiment.evaluation))
				self.saveModels()

	def saveModels(self):
		wrapper = self.getWrapper(getBest=True)
		path = os.path.join(self.experiment.expPath, "model.pth")
		print("path", path)
		torch.save(wrapper.model, path)

	def getWrapper(self, getBest=False):
		wrapper = ModelWrapper(savePath=self.experiment.expPath, device=self.experiment.device, seed=self.seed, printLvl=10)
		wrapper.setModel(self.experiment.model, self.experiment.inputDim, self.experiment.outputDim, self.experiment.modelParams)
		wrapper.setOptimizer(self.experiment.optimizer, learningRate=self.experiment.learningRate)
		wrapper.setCriterion(self.experiment.criterion)
		wrapper.loadCheckpoint()
		if getBest: wrapper.loadBestModel()
		return wrapper


	def trainModel(self):
		print("Training model ...")
		datasetTrain = DataReader(self.experiment.data["path"], onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		datasetTrain.setDatasetClassic("train", self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.testRun: datasetTrain.keepOneOnly()
		datasetDev = DataReader(self.experiment.data["devPath"], onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		datasetDev.setDatasetClassic("dev", self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.experiment.data["featModelPath"] != "" and self.onlineFeat: 
			datasetTrain.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			print("fairseq model for training data loaded")
			datasetDev.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			print("fairseq model for development data loaded")
		if self.testRun: datasetDev.keepOneOnly()
		if self.testRun: self.experiment.maxEpoch = 1
		inp, tar = datasetDev[0]
		self.experiment.inputDim = inp.shape[1]
		if not "classification" in self.experiment.genre:
			self.experiment.outputDim = tar.shape[1]
		# print("experiment.outputDim", tar.shape)
		wrapper = self.getWrapper()
		wrapper.trainModel(datasetTrain, datasetDev, batchSize= self.experiment.batchSize, maxEpoch= self.experiment.maxEpoch, 
						   loadBefore=True, tolerance = self.experiment.tolerance, 
						   minForTolerance= self.experiment.minForTolerance, limitTrainData=self.limitTrainData, limitDevData=self.limitDevData)
		wrapper.saveLogToCSV()
		

	# def writeModelOutput(self):
	# 	# print("Writing outputs ...")
	# 	dataset = DataReader(self.experiment.data["path"], onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
	# 	dataset.setDatasetFeatOnly(self.experiment.testOn, self.experiment.data["feature"])
	# 	if self.experiment.data["featModelPath"] != "": 
	# 		dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"])
	# 		print("fairseq model for writting model outputs loaded")
	# 	if self.testRun: dataset.keepOneOnly()
	# 	dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
	# 	wrapper = self.getWrapper(getBest=True)
	# 	modelOutPath = os.path.join(wrapper.savePath, "outputs")
	# 	if not os.path.exists(modelOutPath): os.makedirs(modelOutPath)
	# 	for idx, (ID, feat) in enumerate(dataloader):
	# 		output = wrapper.forwardModel(feat)
	# 		output = output.detach().cpu().numpy()
	# 		# print(ID, feat.shape, output.shape)
	# 		savePath = os.path.join(modelOutPath, ID[0]+".csv")
	# 		headers = ["output_"+str(i) for i in range(output.shape[2])]
	# 		df = pd.DataFrame(output[0], columns = headers)
	# 		df.to_csv(savePath, index=False)
	# 		printProgressBar(idx+1, len(dataloader), prefix = 'Writing outputs:', suffix = '', length = "fit")


	def writeModelOutput(self):
		# print("Writing outputs ...")
		if "classification" in self.experiment.genre:
			self.writeOutForClassification()
		else:
			self.writeOutForRegression()
	

	def writeOutForRegression(self):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		dataset.setDatasetFeatOnly(self.experiment.testOn, self.experiment.data["feature"])
		if self.experiment.data["featModelPath"] != "" and self.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			print("fairseq model for writting model outputs loaded")
		if self.testRun: dataset.keepOneOnly()
		dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
		wrapper = self.getWrapper(getBest=True)
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

	def writeOutForClassification(self):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		dataset.setDatasetFeatOnly(self.experiment.testOn, self.experiment.data["feature"])
		if self.experiment.data["featModelPath"] != "" and self.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			print("fairseq model for writting model outputs loaded")
		if self.testRun: dataset.keepOneOnly()
		dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
		wrapper = self.getWrapper(getBest=True)
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


	def testModel(self):
		if "classification" in self.experiment.genre:
			return self.testClassification()
		else:
			return self.testRegression()


	def testRegression(self):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		dataset.setDatasetAnnotOnly(self.experiment.testOn, self.experiment.data["annotation"])
		# if self.experiment.data["featModelPath"] != "" and self.onlineFeat: 
		# 	dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"])
		# 	print("fairseq model for testing model outputs loaded")
		idx, tar = dataset[0]
		# experiment.inputDim = inp.shape[1]
		# if not "classification" in experiment.genre:
		self.experiment.outputDim = tar.shape[1]

		firstID1 = list(dataset.dataPart.keys())[0]
		firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
		headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
		# print(headers)
		wrapper = self.getWrapper(getBest=True)
		modelOutPath = os.path.join(wrapper.savePath, "outputs")
		if self.testRun: dataset.keepOneOnly()
		IDs = dataset.dataPart.keys()
		# for key in self.experiment.metrics:
		evaluations = {}
		evaluation = {}
		allTars = []
		allOuts = []
		results = np.zeros((len(self.experiment.metrics), len(headers), len(IDs)))
		for idx, ID in enumerate(IDs):
			savePath = os.path.join(modelOutPath, ID+".csv")
			outputs = pd.read_csv(savePath).to_numpy()
			targets = dataset.targetReader(ID)
			# RESAMPLE OUTPUT TO TARGETS FOR TESTING!
			if self.resampleTarget: 
				from Utils.Funcs import reshapeMatrix
				outputs = reshapeMatrix(outputs, len(targets))
			# print(targets.shape, outputs.shape)
			# if idx == 0:
				#[[] for _ in range(targets.shape[1])]
				# bestresult = 0; bestID = "0"
				# for dim in range(targets.shape[1]): evaluation[headers[dim]] = {} 
			for dim in range(targets.shape[1]):
				output = outputs[:, dim]
				target = targets[:, dim]
				# while target.shape[0] > output.shape[0]: output = np.append(output, outputs[-1])
				# while target.shape[0] < output.shape[0]: output = outputs[:target.shape[0]].reshape(target.shape[0])
				while target.shape[0] != output.shape[0]: output = outputs.reshape(target.shape[0])
				if self.testConcated: allTars+=list(target);  allOuts+=list(output)
				for k, key in enumerate(self.experiment.metrics):
					result = getMetric(target, output, metric=key)
					results[k, dim, idx] = result
				# if result > bestresult: bestresult=result; bestID = ID
				# print(ID, result, len(output))
				
			printProgressBar(idx+1, len(IDs), prefix = 'Testing model:', suffix = '', length = "fit")
		for k, key in enumerate(self.experiment.metrics):
			for dim in range(targets.shape[1]): 
				if self.testConcated: 
					evaluation[headers[dim]] = getMetric(np.array(allTars), np.array(allOuts), metric=key)
					if key=="AUC": # write fpr & tpr to plot ROCs!
						from sklearn import metrics
						fpr, tpr, thresholds = metrics.roc_curve(np.array(allTars), np.array(allOuts))
						fpr =  reshapeMatrix(np.expand_dims(fpr, axis=1), 100)
						tpr =  reshapeMatrix(np.expand_dims(tpr, axis=1), 100)
						savePath = os.path.join(wrapper.savePath, "ROC_resampled_"+str(dim)+"_"+os.path.split(self.currentDataPath)[-1]+".csv")
						np.savetxt(savePath, [np.squeeze(fpr), np.squeeze(tpr)], delimiter=",")
				else:
					evaluation[headers[dim]] = {}
					evaluation[headers[dim]]['mean'] = np.mean(results[k, dim])
					evaluation[headers[dim]]['std'] = np.std(results[k, dim])
			evaluations[key] = evaluation.copy()
			# print("evaluation",evaluation,key)
			# print("self.experiment.evaluation[key]", key, self.experiment.evaluation[key])
		return evaluations


	def testClassification(self):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.onlineFeat, resampleTarget=self.resampleTarget)
		dataset.setDatasetClassic(self.experiment.testOn, self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.experiment.data["featModelPath"] != "" and self.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			print("fairseq model for testing model outputs loaded")
		inp, tar = dataset[0]
		self.experiment.inputDim = inp.shape[1]
		firstID1 = list(dataset.dataPart.keys())[0]
		firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
		headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
		# print(headers)
		wrapper = getWrapper(self.experiment, seed=self.seed, getBest=True)
		modelOutPath = os.path.join(wrapper.savePath, "outputs")
		savePath = os.path.join(modelOutPath, "outputs.csv")
		outputsCSV = pd.read_csv(savePath)
		if self.testRun: dataset.keepOneOnly()
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
		evaluation = {}
		for key in self.experiment.metrics:
			evaluation[key] = getMetric(target, output, metric=key)
		self.experiment.evaluation = evaluation
		confMat = confMatrix(target, output, numTars=self.experiment.outputDim)
		# print(confMatrix(target, output, numTars=experiment.outputDim))
		savePath = os.path.join(wrapper.savePath, "confMat.csv")
		np.savetxt(savePath, confMat, delimiter=",")
		return evaluation


	def getExpIDs(self):
		with open(self.jsonPath, 'r') as jsonFile: 
		    jsonDict = json.load(jsonFile)
		    expIDs = jsonDict.keys()
		return expIDs


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="path to experiments.json file")
    parser.add_argument('--testRun', '-t', help="testing by running only on one data", default=False)
    parser.add_argument('--seed', '-s', help="seed used for random number generation by torch", default=0)
    parser.add_argument('--testConcated', '-c', help="testing by running only on one data", default=False)
    parser.add_argument('--onlineFeat', '-o', help="testing by running only on one data", default=False)
    parser.add_argument('--resampleTarget', '-rt', help="testing by running only on one data", default=False)
    
    args = parser.parse_args()
    Flag = False
    if args.testRun == "False": args.testRun = False
    if args.testConcated == "False": args.testConcated = False
    if args.onlineFeat == "False": args.onlineFeat = False
    if args.resampleTarget == "False": args.resampleTarget = False
    if args.json is None: Flag=True
    if Flag:
        parser.print_help()
    else:
        runner = Runner(args.json, int(args.seed), args.testRun, args.testConcated, args.onlineFeat, args.resampleTarget)
        runner.main()
