# Automatic Emotion Recognition

## Requirements

To satisfy the requirements for running experiments please refer to environment.yml file,

```bash
conda env create -f environment.yml
conda activate emotionCampaign1
```

Note: To extract `wav2vec` features using fairseq, the version used is "1.0.0a0+5273bbb", please check the version, at the time of writing this it is the version on their github and not the one that can be installed by `pip`.

#### In case of failure of the instructions above

---

Create a new environment on conda and after activating it, run the following commands to get the necessary packages installed:

```bash
conda create --name emotionCampaign python=3
conda activate emotionCampaign
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch # Please change cudatoolkit depending on your case!
pip install python_speech_features 
# Follow fairseq installation from: https://github.com/pytorch/fairseq
conda install pandas
conda install -c conda-forge pydub
conda install -c anaconda scipy
```

## Access to Datasets

Datasets used here were RECOLA and AlloSat, information regarding how to access them can be found below:

RECOLA: https://diuf.unifr.ch/main/diva/recola/download.html

AlloSat: https://lium.univ-lemans.fr/en/allosat/



## Running experiments for RECOLA

1- Change directory (`cd`) to `Scripts/DatasetsRunner/RECOLA_2016` 

2- Open the file and set the annotations, models, features, and the path to those features in the `setExperiments` function. It is already filled with some examples as it was used for `LeBenchmark`, so please feel free to change those.

3- Run the file:

```bash
python RECOLA_2016.py -w "[path_to_datasets]/RECOLA_2016/recordings_audio" -a "[path_to_datasets]/RECOLA_2016/ratings_gold_standard" -o "[path_to_datasets]/RECOLA_2016"
```

**IMPORTANT NOTE**: The mentioned file only works with the version of RECOLA provided for AVEC_2016 challenges and it requires the `ratings_gold_standard` and `recordings_audio` to be under `[path_to_datasets]/RECOLA_2016`.
