import os, glob, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from Utils.Funcs import get_files_in_path, printProgressBar
import argparse

def main(inputPath, newPath, ignorePath, ext):
    """
        Transform all audio files under a folder into wav PCM 16kHz 16bits signed-intege.

        example: 
            python Preprocess.py --input "../../Data/Wavs" --output "../../Data/WavsProcessed"
            python Preprocess.py --input "/mnt/HD-Storage/Databases/AlloSat_corpus/audio" --output "/mnt/HD-Storage/Datasets/AlloSat/Wavs"

            python Preprocess.py --input "/mnt/HD-Storage/Databases/ESTER1/ESTER1_TRAIN/wav" --output "/mnt/HD-Storage/Datasets/ESTER1/Wavs/train"
            python Preprocess.py --input "/mnt/HD-Storage/Databases/ESTER1/ESTER1_DEV/wav" --output "/mnt/HD-Storage/Datasets/ESTER1/Wavs/dev"
            python Preprocess.py --input "/mnt/HD-Storage/Databases/ESTER1/ESTER1_TEST/wav" --output "/mnt/HD-Storage/Datasets/ESTER1/Wavs/test"

            python Preprocess.py --input "/mnt/HD-Storage/Databases/RAVDESS_Audio_Speech_Actors_01-24" --output "/mnt/HD-Storage/Datasets/RAVDESS/Wavs"
            python Preprocess.py --input "/mnt/HD-Storage/Databases/Noises4" --output "/mnt/HD-Storage/Datasets/NoiseFiles"

            python Preprocess.py --input "/mnt/HD-Storage/Databases/SEMAINE/wav_data_original" --output "/mnt/HD-Storage/Datasets/SEMAINE/Wavs"
            python Preprocess.py --input "/mnt/HD-Storage/Databases/IEMOCAP/IEMOCAP_full_release" --output "/mnt/HD-Storage/Datasets/IEMOCAP/Wavs" -e "avi"

            python Preprocess.py --input "/mnt/HD-Storage/Databases/MaSS/output_waves" --output "/mnt/HD-Storage/Datasets/MaSS_Fr/Wavs"
    """
    path = os.path.join(inputPath, "**")#"../PMDOM2FR/**/"
    theFiles = get_files_in_path(path, ext=ext)

    for i, filePath in enumerate(theFiles):
        # Making wav files
        fileNewPath = filePath.replace(inputPath, newPath)
        if ignorePath: fileNewPath = os.path.join(newPath, os.path.split(filePath)[-1])
        makeDirFor(fileNewPath)
        if ext=="avi":
            os.system('ffmpeg -i ' + filePath + ' -y ' + "temp.wav")
            os.system('sox ' + "temp.wav" + ' -r 16000 -c 1 -b 16 -e signed-integer ' + fileNewPath[:-4]+".wav")
            os.remove("temp.wav")
        else:
            os.system('sox ' + filePath + ' -r 16000 -c 1 -b 16 -e signed-integer ' + fileNewPath)
        printProgressBar(i + 1, len(theFiles), prefix = 'Transforming Files:', suffix = 'Complete')
    
def makeDirFor(filePath):
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input wavs folder path")
    parser.add_argument('--output', '-o', help="output wavs folder path")
    parser.add_argument('--ignorePath', '-ip', help="ignore the path of the wav files and put all files in the given folder", default=False)
    parser.add_argument('--inputFileExtention', '-e', help="input files extention, default is \"wav\"", default="wav")
    args = parser.parse_args()
    Flag = False
    if args.input is None: Flag=True
    if args.output is None: Flag=True
    if args.ignorePath == "False": args.ignorePath = False
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.input, args.output, args.ignorePath, args.inputFileExtention)