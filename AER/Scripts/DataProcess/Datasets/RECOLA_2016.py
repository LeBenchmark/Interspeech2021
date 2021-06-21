import os, sys, argparse, json
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))
from DataProcess.Preprocess import main as Preprocess
from DataProcess.makeJson import main as makeJson

def main(wavsFolder, annotations, outFolder):
	"""
	example:
		python RECOLA_2016.py -w "/mnt/HD-Storage/Datasets/RECOLA_2016/recordings_audio" -a "/mnt/HD-Storage/Datasets/RECOLA_2016/ratings_gold_standard" -o "/mnt/HD-Storage/Datasets/RECOLA_2016"
	"""
	
	## Preprocess wav files
	print("Preprocess wav files...")
	newFolder = os.path.join(outFolder, "wavs")
	Preprocess(wavsFolder, newFolder, False, "wav")

	## Create json file
	makeJson(outFolder, ["train_*", "dev_*", "test_*"])

	## Create labels

	## add Annots


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', '-a', help="path to folder contatining annotations files", default="")
    parser.add_argument('--wavsFolder', '-w', help="path to folder containing wav files", default="")
    parser.add_argument('--outFolder', '-o', help="path to folder containing output files", default="")
    
    args = parser.parse_args()
    Flag = False
    if args.annotations == "": Flag = True
    if args.wavsFolder == "":  Flag = True
    if args.outFolder == "":   Flag = True
    if Flag:
        parser.print_help()
    else:
        main(args.wavsFolder, args.annotations, args.outFolder)

