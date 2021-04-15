# Automatic Speech-to-text Translation (AST)

We describe the steps to reproduce our AST results presented in the paper (Section 5.3).  

The following table (corresponding to Table 5 in the paper) shows the results using different speech features on the dev/valid and test sets of CoVoST-2 (CV2) and multilingual TEDx (mTEDx).

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="4">Dev/Valid data</th>
      <th colspan="4">Test data</th>
      <th></th>
    </tr>
  </thead>
    <thead>
    <tr>
      <th></th>
      <th>CV2</th>
      <th colspan="3">multilingual TEDx</th>
      <th>CV2</th>
      <th colspan="3">multilingual TEDx</th>
      <th></th>
    </tr>
    </thead>
    <thead>
    <tr>
      <th>Input features</th>
      <th>fr-en</th>
      <th>fr-en</th>
      <th>fr-es</th>
      <th>fr-pt</th>
      <th>fr-en</th>
      <th>fr-en</th>
      <th>fr-es</th>
      <th>fr-pt</th>
      <th>Link</th>
    </tr>
    </thead>
  <tbody>
    <tr>
	 <td>MFB</td>
	 <td><b>23.37</td>
	 <td>1.14</td>
	 <td>0.84</td>
	 <td>0.49</td>
	 <td><b>22.66</td>
	 <td>1.33</td>
	 <td>0.98</td>
	 <td>0.68</td>
	 <td></td>
</tr>
<tr>
	 <td>W2V2-En-base</td>
	 <td>19.24</td>
	 <td>0.90</td>
	 <td>0.65</td>
	 <td>0.43</td>
	 <td>18.19</td>
	 <td>0.88</td>
	 <td>0.34</td>
	 <td>0.27</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt>Download</a></td>
</tr>
<tr>
	 <td>W2V2-En-large</td>
	 <td>17.07</td>
	 <td>0.75</td>
	 <td>0.61</td>
	 <td>0.45</td>
	 <td>16.45</td>
	 <td>0.85</td>
	 <td>0.67</td>
	 <td>0.32</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-S-base</td>
	 <td>19.86</td>
	 <td>2.64</td>
	 <td>0.49</td>
	 <td>0.50</td>
	 <td>19.04</td>
	 <td>1.66</td>
	 <td>0.67</td>
	 <td>0.61</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-base>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-S-large</td>
	 <td>19.62</td>
	 <td>5.12</td>
	 <td>4.62</td>
	 <td><b>2.06</td>
	 <td>18.61</td>
	 <td>2.97</td>
	 <td>3.19</td>
	 <td><b>2.25</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-S-large>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-M-base</td>
	 <td>19.47</td>
	 <td>6.98</td>
	 <td>1.87</td>
	 <td>0.63</td>
	 <td>18.32</td>
	 <td>6.37</td>
	 <td>1.99</td>
	 <td>0.54</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-M-large</td>
	 <td>20.17</td>
	 <td><b>9.35</td>
	 <td><b>7.72</td>
	 <td>1.58</td>
	 <td>19.35</td>
	 <td><b>6.76</td>
	 <td><b>6.63</td>
	 <td>1.63</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-VP-base</td>
	 <td>18.44</td>
	 <td>0.81</td>
	 <td>0.45</td>
	 <td>0.56</td>
	 <td>17.40</td>
	 <td>0.89</td>
	 <td>0.58</td>
	 <td>0.75</td>
	 <td><a href=https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_fr.pt>Download</a></td>
</tr>
<tr>
	 <td>W2V2-Fr-VP-large</td>
	 <td>20.72</td>
	 <td>7.43</td>
	 <td>4.66</td>
	 <td>0.43</td>
	 <td>19.88</td>
	 <td>5.39</td>
	 <td>3.62</td>
	 <td>0.49</td>
	 <td><a href=https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_fr.pt>Download</a></td>
</tr>
<tr>
	 <td>XLSR-53-large</td>
	 <td>20.54</td>
	 <td>0.59</td>
	 <td>0.41</td>
	 <td>0.49</td>
	 <td>19.93</td>
	 <td>0.44</td>
	 <td>0.62</td>
	 <td>0.29</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>Download</a></td>
</tr>
  </tbody>
</table>

<!-- |                    |              Dev/Valid data ||||                   Test data ||||  |
|                    | CoVoST-2 | multilingual TEDx ||| CoVoST-2 | multilingual TEDx |||  |
| Input features     | fr-en | fr-en | fr-es | fr-pt | fr-en | fr-en | fr-es | fr-pt | Link|
| -----              | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| MFB                |   **23.37** | 1.14 | 0.84 | 0.49 | **22.66** | 1.33 | 0.98 | 0.68 | |
| W2V2-En-*base*     |   19.24 | 0.90 | 0.65 | 0.43 | 18.19 | 0.88 | 0.34 | 0.27 | [Download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) |
| W2V2-En-*large*    |   17.07 | 0.75 | 0.61 | 0.45 | 16.45 | 0.85 | 0.67 | 0.32 | [Download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt) |
| W2V2-Fr-S-*base*   |   19.86 | 2.64 | 0.49 | 0.50 | 19.04 | 1.66 | 0.67 | 0.61 | [Download](https://huggingface.co/LeBenchmark/wav2vec2-FR-S-base) |
| W2V2-Fr-S-*large*  |   19.62 | 5.12 | 4.62 | **2.06** | 18.61 | 2.97 | 3.19 | **2.25** | [Download](https://huggingface.co/LeBenchmark/wav2vec2-FR-S-large) |
| W2V2-Fr-M-*base*   |   19.47 | 6.98 | 1.87 | 0.63 | 18.32 | 6.37 | 1.99 | 0.54 | [Download](https://huggingface.co/LeBenchmark/wav2vec2-FR-M-base) |
| W2V2-Fr-M-*large*  |   20.17 | **9.35** | **7.72** | 1.58 | 19.35 | **6.76** | **6.63** | 1.63 | [Download](https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large) |
| W2V2-Fr-VP-*base*  |   18.44 | 0.81 | 0.45 | 0.56 | 17.40 | 0.89 | 0.58 | 0.75 | [Download](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_fr.pt) |
| W2V2-Fr-VP-*large* |   20.72 | 7.43 | 4.66 | 0.43 | 19.88 | 5.39 | 3.62 | 0.49 | [Download](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_fr.pt) |
| XLSR-53-*large*    |   20.54 | 0.59 | 0.41 | 0.49 | 19.93 | 0.44 | 0.62 | 0.29 | [Download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt) | -->

`MFB` means log Mel filterbank, `W2V2` denotes wav2vec models trained on monolingual speech, and `XLSR-53` is the multilingual wav2vec model trained on [53 languages](https://arxiv.org/abs/2006.13979). Except for the 4 models named `W2V2-Fr-S/M` which are the ones that we trained on our collected datasets, all remaining models are off-the-shelf ones that we obtained from previous work.

## Datasets
We selected subsets having French as the source language in two large multilingual speech corpora: CoVoST-2 and multilingual TEDx. 
Our benchmark covers translation directions from French (`fr`) to three target languages: English (`en`), Portugese (`pt`), and Spanish (`es`). 
The training sizes (in hours) are shown in the following table.

| Dataset     | fr-en | fr-es | fr-pt | Link |
| ----- | ----- | ----- | ----- | ----- |
| CoVoST-2    |   180     | -     | -     | [Download](https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/fr.tar.gz) |
| mTEDx       |   50      |  38   | 25    | [Download](https://www.openslr.org/resources/100) |


## Installation
Our implementation is based on [fairseq S2T](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text). 
Please clone [our fork](https://github.com/formiel/fairseq/tree/LeBenchmark) (`LeBenchmark` branch)
as there are modifications made for LeBenchmark:

```bash
git clone https://github.com/formiel/fairseq.git
cd fairseq
git checkout LeBenchmark
```

Then install it in your environment:
```bash
pip install -e . 
```
(remove `-e` in the above if you don't want to install in the editable mode)

## Preprocessing
To extract speech representations from pre-trained wav2vec models for each dataset,
run the following commands:

**CoVoST-2:**
```bash
python examples/speech_to_text/prep_covost_data_w2v_feats.py \
    --data-root ${COVOST_ROOT} \
    --src-lang fr --tgt-lang en \
    --vocab-type char \
    --use-w2v-feats \
    --w2v-path /path/to/pretrained_wav2vec_model
```
**Multilingual TEDx:**
<!-- For multilingual TEDx, you can downsample the audio files to 16kHz using the 
script in this repo:

```bash
python AST/tools/convert_to_wav.py --audio-dir /path/to/original/audio/folder \
                                   --audio-ext flac \
                                   --wav-dir /path/to/output/folder
``` -->

```bash
python examples/speech_to_text/prep_mtedx_data_w2v_feats.py \
    --data-root ${MTEDX_ROOT} \
    --task st \
    --vocab-type unigram --vocab-size 1000 \
    --use-w2v-feats \
    --w2v-path /path/to/pretrained_wav2vec_model
```
where `${COVOST_ROOT}` and `${MTEDX_ROOT}` are paths to the root folders of CoVoST-2 
and multilingual TEDx, repsectively.


## Training
To train a speech_to_text translation model on the extracted features,
run the following command:

```bash
fairseq-train ${DATA_ROOT} \
    --train-subset train_st_fr_en \
    --valid-subset dev_st_fr_en \
    --config-yaml config_st_fr_en.yaml \
    --save-dir ${SAVE_DIR} \
    --max-tokens 40000 \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy \
    --report-accuracy \
    --max-epoch 200 \
    --arch s2t_transformer_s \
    --optimizer adam \
    --lr 2e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --clip-norm 10.0 \
    --seed 1 \
    --log-interval 10 \
    --update-freq 8 \
    --tensorboard-logdir ${TENSORBOARD_DIR} \
    --use-linear-before-cnn \
    --encoder-freezing-updates 0
```
where:
- `${DATA_ROOT}` is `${COVOST_ROOT}/fr` for CoVoST-2 and `${MTEDX_ROOT}/fr-en` 
for multilingual TEDx.
- `${SAVE_DIR}` and `${TENSORBOARD_DIR}` are the paths to where you want to save 
the trained model and the tensorboard plots, respectively.

## Decoding
To decode using a trained model (with weight-averaging over the last 10 checkpoints),
run the following commands:

```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} \
    --num-epoch-checkpoints 10 \
    --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${DATA_ROOT} \
    --config-yaml ${CONFIG} \
    --gen-subset ${GEN_SUBSET} \
    --task speech_to_text \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring sacrebleu \
    --results-path ${RESULT_PATH}
```
where:
- `${CONFIG}` is `config_st_fr_en.yaml` for CoVoST-2 and `config_st.yaml` for 
multilingual TEDx.
- `${GEN_SUBSET}` is the name of the subset you want to decode.
- `${RESULT_PATH}` is the path where you want to save the decoding results.
