# Spoken Language Understanding (SLU)

The SLU Benchmark used for the **LeBenchmark** is **MEDIA** ([See here for downloading input features](http://www.marcodinarelli.it/is2021.php)).

The system used is based on Sequence-to-Sequence models and is coded for the [Fairseq library](https://github.com/pytorch/fairseq).
The encoder is similar to the pyramidal LSTM-based encoder proposed in the [Listen, attend and spell paper](https://arxiv.org/abs/1508.01211) (Kheops), the only difference is that we compute the mean of two consecutive hidden states for reducing the output size between two layers, instead of concatenating them like in the original model.
The decoder is similar to the one used in our previous work published at [ICASSP 2020](http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf). The differences in this case are that we use two attention mechanisms, one for attending the encoder hidden states, and the other for attending all the previous predictions, instead of using only the previous one like in the original decoder.

We use a similar training strategy as in [our previous work](http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf).
We train thus the encoder alone first, by putting a simple decoder (Basic) on top of it, that is a linear layer mapping the encoder hidden states into the output vocabulary size. The pre-trained encoders are used to pre-initialize parameters of models using a LSTM decoder (LSTM).
Models with a Basic decoder and trained for decoding tokens (ASR) are used to pre-initialize models with a Basic decoder trained for SLU.
Results obtained with this strategy are summarized in the following table, we give both token decoding (ASR) and concept decoding (SLU) results.
For more details please see the [paper accepted at Interspeech 2021](https://arxiv.org/abs/2104.11462).

<center>
<table>
  <thead>
    <tr>
      <th colspan="4"> Token decoding (Word Error Rate)</th>
    </tr>  
    <tr>
      <th> Model </th>
      <th> Input Features </th>
      <th> DEV ER </th>
      <th> TEST ER </th>
    </tr>
  </thead>
  
  <tbody>
    <thead>
      <tr>
        <th colspan="4"> Comparison to our previous work </th>
      </tr>
    </thead>
    <tr>
      <td> <a href ="http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf">ICASSP 2020 Seq</a> </td> <td> Spectrogram </td> <td> 29.42 </td> <td> 28.71 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> Interspeech 2021 </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> Spectrogram </td> <td> 36.25 </td> <td> 37.16 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt>W2V2-En-base</a> </td> <td> 19.80 </td> <td> 21.78 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a> </td> <td> 24.44 </td> <td> 26.96 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-base>W2V2-Fr-S-base</a> </td> <td> 23.11 </td> <td> 25.22 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-large>W2V2-Fr-S-large</a> </td> <td> 18.48 </td> <td> 19.92 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a> </td> <td> 14.97 </td> <td> 16.37 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large>W2V2-Fr-M-large</a> </td> <td> <b>11.77</b> </td> <td> <b>12.85</b> </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR53-large</a> </td> <td> 14.98 </td> <td> 15.74 </td>
    </tr>
  </tbody>
  
  <thead>
    <tr>
      <th colspan="4"> Concept decoding (Concept Error Rate)</th>
    </tr>  
    <tr>
      <th> Model </th>
      <th> Input Features </th>
      <th> DEV ER </th>
      <th> TEST ER </th>
    </tr>
  </thead>
  
  <tbody>
    <thead>
      <tr>
        <th colspan="4"> Comparison to our previous work </th>
      </tr>
    </thead>
    <tr>
      <td> <a href ="http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf">ICASSP 2020 Seq</a> </td> <td> Spectrogram </td> <td> 28.11 </td> <td> 27.52 </td>
    </tr>
    <tr>
      <td> <a href ="http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf">ICASSP 2020 XT</a> </td> <td> Spectrogram </td> <td> 23.39 </td> <td> 24.02 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> Interspeech 2021 </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> Spectrogram </td> <td> 39.66 </td> <td> 40.76 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> Spectrogram </td> <td> 34.38 </td> <td> 34.74 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> Spectrogram </td> <td> 33.63 </td> <td> 34.76 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt>W2V2-En-base</a> </td> <td> 26.79 </td> <td> 26.57 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt>W2V2-En-base</a> </td> <td> 26.31 </td> <td> 26.11 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a> </td> <td> 29.31 </td> <td> 30.39 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a> </td> <td> 28.38 </td> <td> 28.57 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-base>W2V2-Fr-S-base</a> </td> <td> 27.18 </td> <td> 28.27 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-base>W2V2-Fr-S-base</a> </td> <td> 26.16 </td> <td> 26.69 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-large>W2V2-Fr-S-large</a> </td> <td> 23.34 </td> <td> 23.75 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-large>W2V2-Fr-S-large</a> </td> <td> 22.53 </td> <td> 23.03 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a> </td> <td> 22.11 </td> <td> 21.30 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>W2V2-Fr-M-base</a> </td> <td> 22.56 </td> <td> 22.24 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large>W2V2-Fr-M-large</a> </td> <td> 21.72 </td> <td> 21.35 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large>W2V2-Fr-M-large</a> </td> <td> <b>18.54</b> </td> <td> <b>18.62</b> </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic +token </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR53-large</a> </td> <td> 21.00 </td> <td> 20.67 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> <a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR53-large</a> </td> <td> 20.34 </td> <td> 19.73 </td>
    </tr>
    
  </tbody>
  
 </table>
</center>

# Installation

The system was developped under **python 3.7, pytorch 1.4.0 and Fairseq 0.9**, it can probably work with other version of python and pytorch, but it will not work for sure under Fariseq 0.10 or more recent. Tha's why we recommend, in order to reproduce our results, to download and install [this version of Fairseq 0.9](https://github.com/pytorch/fairseq/tree/d5618e9ecd05a1edaba5c0991efb3503f121d132).
Once you have a running installation of Fairseq, you just have to copy files in the correct directories:

- **End2EndSLU.py** in fairseq/tasks/
- **End2EndSLUDataset.py** in fairseq/data/
- **End2EndSLUModels.py** in fairseq/models/
- **SLU_CTC_loss.py** in examples/speech_recognition/criterions/ (Fairseq 0.9)
- **globals.py** in fairseq/
- **slu_models.py** in fairseq/

**compute_error_rate.py** is used from command line (see Usage below) to compute the error rate on the model output.
**extract_flowbert_features.py** is used from command line (see Usage below) to extract features from wav signals with wav2vec 2.0 models. **NOTE**: for using this script Fairseq 0.10 or more recent is needed (you can install different versions of Fairseq in different virtual env), as wav2vec 2.0 has been deployed with such Fairseq version.

# Usage

### Features generation

Input features used for our [Interspeech 2021 paper](https://arxiv.org/abs/2104.11462), and for our NeurIPS submission are available [here](http://www.marcodinarelli.it/is2021.php), so that you don't need the original data or to extract features on your own.

If you want to extract features on your own with your wav2vec 2.0 models, you can use the **extract_flowbert_features.py** script.
Since features are extracted once for all, I did not add command line options to the script, you need to modify flags and variables in the script.
Some models used to extract features are reachable via links in the table above, or from our [HuggingFace repository](https://huggingface.co/LeBenchmark).

Flags:
- **upsample**: if set to True, the script will upsample signals to twice the sample rate (this is because MEDIA is 8kHz but 16kHz signals are needed)
- **cuda**: if set to True, the script will use a GPU for extracting features. This is much faster, but MEDIA contains long signals that make the script crash, so for some of them I run on CPU.
- **extract_features**: if set to True, the script will extract features, otherwise it will upsample signals and save them if *upsample* is set to True, otherwise the script will just show info all signals (e.g. duration)
- **model_size**: if set to 'large', the script will perform a layer normalization on input signals, as needed with *large* wav2vec 2.0 models
- **add_ext**: if set to True, the script will assume the input list is made of filename prefixes without extension, and will add it if needed
- **file_ext**: the file extension to give to the extracted feature files

**NOTE**: you may need to modify also the *device* variable in case you want to run the script on a different GPU than specified in the script.

Variables:
- **prefix_list**: the input list, one file per line with absolute path, with or without extension. If the extension is not given, the script will assume '.wav' as the signal extension
- **flowbert_path**: the absolute path to the wav2vec 2.0 model to use for extracting features. **NOTE**: if you want to extract features with a model finetuned with the [supervised finetuning procedure](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md#fine-tune-a-pre-trained-model-with-ctc), because of the way Fairseq instantiate and load models, you will need to specify also the model used as fine-tuning starting point in the variable **sv_starting_point**. Since in the end Fairseq initialize parameters with the model specified in **flowbert_path**, the second model can be identical to the first.

Once flags and variables have been set properly, you can run the script simply as **python extract_flowbert_features.py** from command line, making sure the correct python environement with Fairseq 0.10 is active.

### Training

In order to train a model with a Basic decoder (a linear layer), run the script **run_end2end_slu_train_basic.sh**.

You need to modify environment variables in the script so that to match installation, home, input, etc. on your machine.

In particular:
- the line **source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate** must be changed to activate your python virtual env
- **FAIRSEQ_PATH** must set to point the folder where the Fairseq tools are located (fairseq-train, fairseq-generate, etc.)
- **WORK_PATH** must be set to any working directory on your machine
- **DATA_PATH** must be set to the directory containing the input data. **DATA_PATH** can be any directory, even not the one containing the data, if you have already serialized input features (see below). In particular your data must be contained in three sub-folders of **DATA_PATH** named **train**, **dev** and **test**. Additionally **DATA_PATH** must contain three files named **train_prefixes.lst**, **dev_prefixes.lst** and **test_prefixes.lst** containing the list, with absolute path, of file prefixes (i.e. without extension) of files for the three splits used by the system. The system will expect to find at least 3 different files for each prefix: a feature file (e.g. spectrogram, wav2vec 2.0 features, etc.) with extension specified with the **--feature-extension** option, a *.txt* file containing the transcription of the audio file, and a *.sem* file containing the transcription semantic annotation. Semantic annotation can be one of two formats which are implicitly specified with the option **--corpus-name**: *media*, or *fsc* (Fluent Speech Command). With *fsc* the content of the *.sem* file will be simply split on white spaces, and each token will be assumed as a concept to be predicted.
- **SERIALIZED_CORPUS** is the common prefix to all serialized input features files (train, dev, test and dict), see [here](http://www.marcodinarelli.it/is2021.php).

The **SERIALIZED_CORPUS** prefix is actually automatically created based on a couple of other variables specified right above in the script: **FEATURES_TYPE**, **FEATURES_SPEC**, **FEATURES_LANG**, **NORMFLAG**.

- **FEATURES_TYPE** can be ***spectro*** for spectrograms, ***FlowBERT*** for wav2vec 2.0 large features, ***FlowBBERT*** for wav2vec 2.0 base features, and son on. See the script itself and/or file names [here](http://www.marcodinarelli.it/is2021.php).
- **FEATURES_SPEC** is used to specify additional information in the file name, e.g. 3Kl for the French wav2vec 2.0 large model trained on 3K hours of speech.
- **FEATURES_LANG** is used for the language of the wav2vec 2.0 model (use ***ML*** for the XLSR53 model features)
- **NORMFLAG** is set always to ***Normalized***.

Note that the values of these variables is completly arbitrary. However they must match those used when creating serialized feature files if you want to (re-)use serialized feature files like those available [here](http://www.marcodinarelli.it/is2021.php).

There are 3 additional variables in the script in the same section: **FEATURES_EXTN**, **SUBTASK**, and **CORPUS**.

- **FEATURES_EXTN** is the file extension of feature files. This is used when you don't have (yet) serialized feature files and you want to read-in input data
- **SUBTASK** can be ***token*** or ***concept*** and is used to specify which units the model should be trained to predict: ***token*** for ASR, ***concept*** for SLU. See also below.
- **CORPUS** is an ID for the corpus used in the experiments. You should leave it set to ***media*** if you are using the MEDIA corpus.

Pay attention to the option **--slu-subtask**: with a value **'token'** you will train an ASR model (token decoding); with a value **'concept'** you will train a SLU model where the expected output format is **SOC** <img src="https://render.githubusercontent.com/render/math?math=w_1^1 \dots w_N^1 C_1"> **EOC** ... **SOC** <img src="https://render.githubusercontent.com/render/math?math=w_1^M \dots w_N^M C_M"> **EOC**.
**SOC** and **EOC** are start and end of concept markers, <img src="https://render.githubusercontent.com/render/math?math=w_1^i \dots w_N^i C_i"> are the words instantiating the concept <img src="https://render.githubusercontent.com/render/math?math=C_i">, followed by the concept itself.

In order to train a model with a LSTM decoder (the version of LSTM decoder described above), run the script **run_end2end_slu_train_icassplstm.sh**.
Again, you need to modify environment variables in the script so that to match installation, home, etc. on your machine. See above.
In this script also you need to set properly the option **--slu-subtask**:

In order to train a model pre-initializing parameters with a previously trained model, use the option:
```--load-fairseq-encoder <model file>```

This option is intended to pre-initilize the encoder as explained in the paper. However the system detects automatically if the decoder's type is the same in the instantiated and loaded models, and in that case it pre-initializes also the decoder. This is especially useful when pre-initializing a SLU model with a linear decoder with a ASR model with a linear decoder.
Pay attention to remove this option when training an ASR model from scratch.

At the first run, the system will read data and save them in a serialized format, containing all the tensors needed for training (and generation). At following runs you can use such data with the option **--serialized-corpus \<data prefix>**. _\<data prefix\>_ is the prefix in common to all the generated files (train, validation, test data plus the dictionary). See also the **SERIALIZED_CORPUS** variable above.
This makes data loading much faster, especially when using _wav2vec_ features as input.

Input features are available [here](http://www.marcodinarelli.it/is2021.php), so that you don't need the original data or to extract features on your own.

### Generation

```run_end2end_slu_test.sh <checkpoint path> <serialized corpus> <sub-task>```

This will generate an output in the same folder as the checkpoint. Once again, you need to set properly some environment variable in the script like for training scripts.

Pay attention to the **\<\<sub-task\>\>** argument which will initialize the **--slu-subtask** option in the script to generate the correct reference to compare the system output with.

### Scoring

```score_model_output.sh <generated output>```

Pay attention to the options:
- **--clean-hyp** this will clean the outputs generated by the system, when it is trained with the CTC loss. If you train the system with another loss, this option is not needed. All results above are obtained training with the CTC loss.
- **--slu-out** when this option is given, the evaluation script (**compute_error_rate.py**) will remove concept boundaries and tokens, keeping only concepts (see the output format above), evaluating thus the model with **Concept Error Rate** (CER). If this option is not given, the output will be evaluated as it is, which may be useful when evaluating ASR models trained with **--slu-subtask token**.

# Citation

Please cite from the Bibtex provided with our [arXiv pre-print](https://arxiv.org/abs/2104.11462).
