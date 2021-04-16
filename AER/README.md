# Automatic Emotion Recognition

## Table of Contents  
  * [Requirements](#Requirements)
  * [Access to Datasets](#Access)
  * [Preprocessing Data](#Preprocessing)
  * [Extracting features](#Features)
    * [Mel Filter Bank (MFB) Features](#MFB)
    * [Wav2vec Features](#Wav2vec)
  * [Running Experiments](#Experiments)
  * [Results](#Results)


<a name="Requirements"></a>
## Requirements

To satisfy the requirements for running experiments please refer to environment.yml file,

```bash
conda env create -f environment.yml
conda activate emotionCampaign1
```

Note: To extract `wav2vec` features using fairseq, the version used is "1.0.0a0+5273bbb"


<a name="Access"></a>
## Access to Datasets

Datasets used here were RECOLA and AlloSat, information regarding how to access them can be found below:

RECOLA: https://diuf.unifr.ch/main/diva/recola/download.html

AlloSat: https://lium.univ-lemans.fr/en/allosat/


<a name="Preprocessing"></a>
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


<a name="Features"></a>
## Extracting features

<a name="MFB"></a>
### Mel Filter Bank (MFB) Features

To produce MFB features, the following code may be used:

```bash
python MelFilterBank.py -f "MFB" -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

Then it should be followed by a standardisation step:

```bash
python Standardize.py -f MFB -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

<a name="Wav2vec"></a>
### Wav2vec Features

An example code to produce Wav2vec features:

```bash
python wav2vec2.py -f "FlowBERT_2952h_base_cut30" -d 29.99 -n True -j "[path_to_datasets]/Datasets/RECOLA/data.json" -m "[path_to_wav2vec_models]/Models/FlowBERT_2952h_base.pt"
```


<a name="Experiments"></a>
## Running Experiments

In order to run experiments, first we need to define it. A script defining the preliminary experiments is provided under `Experiments` folder:

```bash
python setupExp_RECOLA_AlloSat.py
```

This will produce a `experiments.json` file that has the information needed to run a series of experiments. Then, the experiments can be run as follows: 

```bash
python Experimenter.py -j "[predefined_path]/experiments.json"
```

<a name="Results"></a>
## Results

The table below represents the results of the experiments. The numbers are Concordance Correlation Coefficient (CCC) of emotion predictions on the RECOLA and AlloSat test sets.

<table>
  <thead>
    <tr>
      <th colspan="2">Corpus</th>
      <th colspan="2">RECOLA</th>
      <th colspan="1">AlloSat</th>
    </tr>
  </thead>
  <thead>
    <tr>
      <th>Model</th>
      <th>Feature</th>
      <th>Arousal</th>
      <th>Valence</th>
      <th>Satisfaction</th>
    </tr>
  </thead>
  
  <tbody>
   <tr>
    <td>Linear-Tanh</td>
    <td>MFB</td>
    <td>.192</td>
    <td>.075</td>
    <td>.065</td>
   </tr>
   <tr>
    <td>Linear-Tanh</td>
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a></td>
    <td><b>.385</td>
    <td><b>.090</td>
    <td><b>.193</td>
   </tr>
   <tr>
    <td>Linear-Tanh</td>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>.155</td>
    <td>.024</td>
    <td>.093</td>
    
   </tr>
   <tr style="border-bottom:1px solid black">
    <td colspan="100%"></td>
   </tr>
   
   <tr>
    <td>GRU-32</td>
    <td>MFB</td>
    <td>.654</td>
    <td>.252</td>
    <td>.437</td>
   </tr>
   <tr>
    <td>GRU-32</td>
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a></td>
    <td><b>.767</td>
    <td><b>.376</td>
    <td><b>.507</td>
   </tr>
   <tr>
    <td>GRU-32</td>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>.605</td>
    <td>.320</td>
    <td>.446</td>
 
   </tr>
   <tr style="border-bottom:1px solid black">
    <td colspan="100%"></td>
   </tr>

   <tr>
    <td>GRU-64</td>
    <td>MFB</td>
    <td>.712</td>
    <td>.307</td>
    <td>.400</td>
   </tr>
   <tr>
    <td>GRU-64</td>
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a></td>
    <td><b>.760</td>
    <td><b>.352</td>
    <td><b>.507</td>
   </tr>
   <tr>
    <td>GRU-64</td>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>.585</td>
    <td>.280</td>
    <td>.434</td>
   
  </tbody>
</table>
