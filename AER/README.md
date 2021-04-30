# Automatic Emotion Recognition

## Table of Contents  
  * [Requirements](#Requirements)
  * [Access to Datasets](#Access)
  * [Preprocessing Data](#Preprocessing)
  * [Features Extraction](#Features)
    * [Mel Filter Bank (MFB) Features](#MFB)
    * [Wav2vec Features](#Wav2vec)
  * [Running Experiments](#Experiments)
  * [Results](#Results)


<a name="Requirements"></a>
## Requirements

Dependencies required for running experiments can easily be prepared using the environment.yml file,

```bash
conda env create -f environment.yml
conda activate emotionCampaign1
```

Note: To extract `wav2vec` features using fairseq, the version used is "1.0.0a0+5273bbb"


<a name="Access"></a>
## Access to Datasets

Please follow the instructions given on the RECOLA and AlloSat websites in order to have access to data:

RECOLA: https://diuf.unifr.ch/main/diva/recola/download.html

AlloSat: https://lium.univ-lemans.fr/en/allosat/


<a name="Preprocessing"></a>
## Preprocessing Data

Audio files must be converted to 16 kHz, 16 bits single interger, mono channel to extract `wav2vec` representations. This can be achieved using the `Preprocess.py` script:

```bash
python Preprocess.py --input "[path to datasets]/audio" --output "[path to datasets]/AlloSat/Wavs"
```

Once done, descriptive json files need to be created in order to run experiments on the RECOLA and AlloSat datasets, using the `Recola_46.py` and `Allosat.py` scripts in `FeatureExtraction\DatasetHandling`. Each script produces a `data.json` file that includes all the information needed to run experiments on each dataset. 

Once those preprocessing steps achieved, the data structure should be as follows:

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

Note: Please make sure that the path given in each `data.json` file refers to the directory where datasets are stored.

Note: Others audio preprocessing scripts can be found in `FeatureExtraction\DatasetHandling`.

<a name="Features"></a>
## Features extraction

<a name="MFB"></a>
### Mel Filter Bank (MFB) Features

To produce MFB features, which serve as baseline, the following command can be used:

```bash
python MelFilterBank.py -f "MFB" -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

Followed by a standardisation step:

```bash
python Standardize.py -f MFB -j "[path_to_datasets]/Datasets/RECOLA/data.json"
```

<a name="Wav2vec"></a>
### Wav2vec Features

Given a pre-trained model, Wav2vec features can be obtained using the following command:

```bash
python wav2vec2.py -f "FlowBERT_2952h_base_cut30" -d 29.99 -n True -j "[path_to_datasets]/Datasets/RECOLA/data.json" -m "[path_to_wav2vec_models]/Models/FlowBERT_2952h_base.pt"
```


<a name="Experiments"></a>
## Running Experiments

To setup the experiments, a script defining the configurations shall be run first in the `Experiments` folder:

```bash
python setupExp_RECOLA_AlloSat.py
```

This will produce a `experiments.json` file that contains all the information needed to reproduce the series of experiments presented in the paper, using the following command: 

```bash
python Experimenter.py -j "[predefined_path]/experiments.json"
```

<a name="Results"></a>
## Results

Performance is evaluated with the Concordance Correlation Coefficient (CCC) as emotion recognition is performed on time- and value-continuous dimensions (arousal, valence - RECOLA, and satisfaction - AlloSat). The table below reproduces the results of the experiments on the test partition.

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
