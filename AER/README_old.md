# Automatic Emotion Recognition

[TOC]

## Requirements

To satisfy the requirements for running experiments please refer to environment.yml file,

```bash
conda env create -f environment.yml
conda activate emotionCampaign1
```

Note: To extract `wav2vec` features using fairseq, the version used is "1.0.0a0+5273bbb", please check the version, at the time of writing this it is the version on their github and not the one that can be installed by `pip`.



## Access to Datasets

Datasets used here were RECOLA and AlloSat, information regarding how to access them can be found below:

RECOLA: https://diuf.unifr.ch/main/diva/recola/download.html

AlloSat: https://lium.univ-lemans.fr/en/allosat/



## Preprocessing Data

The audio files in each dataset should first be preprocessed to be 16 KHz, mono channel and 16 bit single integer. In order to do that `Preprocess.py` file may be used, an example can be seen here:

```bash
python Preprocess.py --input "[path to datasets]/audio" --output "[path to datasets]/AlloSat/Wavs"
```

Then, under `FeatureExtraction\DatasetHandling`, `Recola_46.py` and `Allosat.py` are files related to produce a `data.json` file that includes all the information needed regarding each dataset to run experiments. 

After all the preprocessing steps, they should be in the following format:

```
. Datasets
+-- RECOLA
|	+-- Wavs
|	+-- Annots
|	data.json
+-- AlloSat
|	+-- Wavs
|	+-- Annots
|	data.json
```

Note: Please change the path inside each file to refer to the directory in which each dataset is stored.

Note: Other preprocessing related files like changing the sampling frequency of the data can be found under `FeatureExtraction\DatasetHandling`.



## Extracting features

### Mel Filter Bank (MFB) Features

To produce MFB features, the following code may be used:

```bash
python MelFilterBank.py -f "MFB" -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

Then it should be followed by a standardisation step:

```bash
python Standardize.py -f MFB -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

### Wav2vec Features

An example code to produce Wav2vec features:

```bash
python wav2vec2.py -f "FlowBERT_2952h_base_cut30" -d 29.99 -n True -j "[path_to_datasets]/Datasets/RECOLA/data.json" -m "[path_to_wav2vec_models]/Models/FlowBERT_2952h_base.pt"
```



## Running Experiments

In order to run experiments, first we need to define it. A script defining the preliminary experiments is provided under `Experiments` folder:

```bash
python setupExp_RECOLA_AlloSat.py
```

This will produce a `experiments.json` file that has the information needed to run a series of experiments. Then, the experiments can be run as follows: 

```bash
python Experimenter.py -j "[predefined_path]/experiments.json"
```

