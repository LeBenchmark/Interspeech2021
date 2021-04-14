# Automatic Speech-to-text Translation (AST)

We describe the steps to reproduce our AST results presented in the paper (Section 5.3).

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