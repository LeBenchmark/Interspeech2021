#!/bin/bash

# TO RUN ON DECORE AND OTHER SERVERS UNCOMMENT THE FOLLOWING LINES
#export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${HOME}/anaconda3/

# TO RUN ON THE LIG GRID WITH OAR UNCOMMENT THE FOLLOWING LINES
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/

export RNNTAGGERPATH=${HOME}/work/tools/Seq2Biseq_End2EndSLU/

echo " ---"
echo "Using python: `which python`"
echo "Using fairseq-train: `which fairseq-train`"
echo " ---"

# -----------------
FEATURES_TYPE='XLSR53'	# Use 'spectro', 'normspectro' or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' ... see below
FEATURES_SPEC='-SV'		# Choose a meaningful infix to add into file names, you can leave it empty for spectrograms (*-spg)
FEATURES_EXTN='.xlsr53-sv'	# Feature file extension, e.g. '.20.0ms-spg', '.bert-3kb', '.bert-3kl', '.bert-3klt', ...
FEATURES_LANG='ML'		# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
SUBTASK='concept'
CORPUS='media'
# -----------------
WORK_PATH=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
#DATA_PATH=${HOME}/work/data/ETAPE/
SERIALIZED_CORPUS=${WORK_PATH}/system_features/MEDIA.user+machine.${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}.data

# LSTMs in the encoder are bi-directional, thus the decoder size must be twice the size of the encoder
encoder_size=$((256))
decoder_size=$((${encoder_size}*2))
ENC_FFN_DIM=$((${decoder_size}*2))
NCONV=1
NLSTM=3
ATT_HEADS=2
ELAYERS=1
DLAYERS=3
DROP_RATIO=0.5
TRANS_DROP_RATIO=0.12
MAX_EPOCHS=150
LR=0.0005
LR_SHRINK=0.98
START_ANNEAL=1
if [[ ${START_ANNEAL} -eq 1 ]]; then
	LR=`echo ${LR} ${LR_SHRINK} | awk '{print $1/$2}'`
fi
WDECAY=0.0001
BATCH=10
MAX_TOKENS=4750
# ----------------
NUM_FEATURES=81
WINTIME=20
if [[ ${FEATURES_TYPE} == 'normspectro' ]]; then
	NUM_FEATURES=1025
fi
if [[ ${FEATURES_TYPE} == 'W2V' ]]; then
	NUM_FEATURES=512
	WINTIME=-1
fi
if [[ ${FEATURES_TYPE} == 'W2V2-Large' ]]; then
	NUM_FEATURES=1024
	WINTIME=-6
fi
if [[ ${FEATURES_TYPE} == 'W2V2' ]]; then
	NUM_FEATURES=768
	WINTIME=-4
fi
if [[ ${FEATURES_TYPE} == 'FlowBERT' ]]; then
	NUM_FEATURES=1024
	WINTIME=-3
fi
if [[ ${FEATURES_TYPE} == 'FlowBERTS' ]]; then
	NUM_FEATURES=2665
	WINTIME=-3
fi
if [[ ${FEATURES_TYPE} == 'FlowBBERT' ]]; then
	NUM_FEATURES=768
	WINTIME=-5
fi
if [[ ${FEATURES_TYPE} == 'XLSR53' ]]; then
	NUM_FEATURES=1024
	WINTIME=-7
fi
# ----------------
CLIP_NORM=5.0
DECODER='basic'
CRITERION='slu_ctc_loss' # cross-entropy doesn't make sense with the basic decoder.

if [[ $# -ge 1 ]]; then
	LR=$1
fi

SAVE_PATH=Ziggurat_Dec-${DECODER}-${SUBTASK}_${CORPUS}_Drop${DROP_RATIO}_${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}_BATCH${BATCH}-LR${LR}-Shrink${LR_SHRINK}-StartAnneal${START_ANNEAL}_forAVG5_LatestCorrect_TEST/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

# Basic token models (for SLU)
um_spg_token_decoder_file='./IS2021/experiments/Ziggurat_Dec-basic-token_media_Drop0.5_spectro-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_TEST/checkpoint.best_avg5.pt'
um_w2v2fr1Kbase_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBBERT-1Kb-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST//checkpoint.best5_avg.pt'
um_w2v2fr1Klarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-1Kl-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_w2v2fr2o6Klarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBBERT-2.6Kb-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt' # DEV WER: 14.22%
um_w2v2fr3Kbase_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBBERT-3Kb-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_w2v2fr3Klarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-3Kl-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt' #'./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-3Kl-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_TEST/checkpoint.best_avg5.pt' # DEV WER 11.05%
um_w2v2fr3Klarge_sstuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-3Klt-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_w2v2fr3Klarge_svtuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-3Kls-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST//checkpoint.best5_avg.pt' # DEV WER: 9.21%
um_w2v2fr7Kbase_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBBERT-7Kb-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt' #'./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBBERT-7Kb-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_NewLoad_TEST/checkpoint.best_avg5.pt' # DEV WER: 15.09%
um_w2v2fr7Klarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-7Kl-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt' #'./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_NewLoad_TEST/checkpoint.best5_avg.pt' # DEV WER: 10.21%
um_w2v2fr7Klarge_sstuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-7Klt-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt' # DEV WER: 10.65%
um_w2v2fr7Klarge_svtuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-7Kls-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST//checkpoint.best5_avg.pt' #'./Ziggurat_Dec-basic-token_media_Drop0.5_FlowBERT-7Kls-Fr-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST//checkpoint.best5_avg.pt' # DEV WER: 9.22%

um_w2v2en1Kbase_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_W2V2-En-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_w2v2en1Klarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_W2V2-Large-En-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'

um_xlsr53mllarge_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_XLSR53-ML-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_xlsr53mllarge_sstuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_XLSR53-SS-ML-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'
um_xlsr53mllarge_svtuned_token_decoder_file='./Ziggurat_Dec-basic-token_media_Drop0.5_XLSR53-SV-ML-Normalized_BATCH10-LR0.000510204-Shrink0.98-StartAnneal1_forAVG5_LatestCorrect_TEST/checkpoint.best5_avg.pt'

# Add the option --save-dir <PATH>, where path contains the checkpoint_last.pt file, in order to load the checkpoint and resume training the model.
# Note: this is a trick to train first a model for characters, then a model for tokens starting from the model for characters, etc.
# ADD: e.g. --load-encoder ${BASIC_MODELS_PATH}/${um_spg_char_decoder_file} to load a pre-trained encoder

warmup_opt="" #"--warmup-updates 5393"
#CUDA_VISIBLE_DEVICES=1
PYTHONPATH=${HOME}/work/tools/fairseq/ ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} --corpus-name ${CORPUS} --feature-extension ${FEATURES_EXTN} --load-fairseq-encoder ${um_xlsr53mllarge_svtuned_token_decoder_file} \
	--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 \
	--decoder ${DECODER} --padded-reference \
	--save-dir ${SAVE_PATH} --patience 20 --no-epoch-checkpoints \
	--speech-conv ${NCONV} --num-features ${NUM_FEATURES} --speech-conv-size ${encoder_size} --drop-ratio ${DROP_RATIO} \
	--num-lstm-layers ${NLSTM} --speech-lstm-size ${encoder_size} --window-time ${WINTIME} --w2v-language ${FEATURES_LANG} \
	--encoder-normalize-before --encoder-layers ${ELAYERS} --encoder-attention-heads ${ATT_HEADS} --encoder-ffn-embed-dim ${ENC_FFN_DIM} \
	--decoder-normalize-before --attention-dropout ${TRANS_DROP_RATIO} --activation-dropout ${TRANS_DROP_RATIO} \
	--decoder-layers ${DLAYERS} --share-decoder-input-output-embed \
	--encoder-embed-dim ${encoder_size} --encoder-hidden-dim ${encoder_size} --decoder-embed-dim ${encoder_size} --decoder-hidden-dim ${encoder_size} \
	--dropout ${TRANS_DROP_RATIO} --decoder-attention-heads ${ATT_HEADS} \
	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} \
	--lr-scheduler fixed --force-anneal ${START_ANNEAL} --lr-shrink ${LR_SHRINK} \
	--optimizer adam --lr ${LR} --clip-norm ${CLIP_NORM} --weight-decay ${WDECAY} ${warmup_opt} \
	--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum 1 --keep-best-checkpoints 5

#deactivate
#conda deactivate


