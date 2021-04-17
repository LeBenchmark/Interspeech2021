#!/bin/bash

# Modify the following variables based on your installation paths
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/


# -----------------
FEATURES_TYPE='spectro'	# Use 'spectro', or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' (See below)
FEATURES_LANG='Fr'	# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
# -----------------
# If you use a serialized version of the corpus, DATA_PATH can be anything, it will not be used
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
SERIALIZED_CORPUS=MEDIA.user+machine.${FEATURES_TYPE}-${FEATURES_LANG}-${NORMFLAG}.data

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
MAX_EPOCHS=100
LR=0.0005
LR_SHRINK=0.98
START_ANNEAL=1
if [[ ${START_ANNEAL} -eq 1 ]]; then
	LR=`echo ${LR} ${LR_SHRINK} | awk '{print $1/$2}'`
fi
WDECAY=0.0001
BATCH=5
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
SUBTASK='token'
CORPUS='media'

if [[ $# -ge 1 ]]; then
	LR=$1
fi

SAVE_PATH=${corpus}_Kheops-${DECODER}_${SUBTASK}_Loss${CRITERION}/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

# Basic token models (for SLU)
um_spg_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_spectro-Fr-Normalized_BATCH5-LR0.0005-Shrink0.98-StartAnneal2_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 36.25%
um_w2v2enbase_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_NormSig-Web-W2V2-En-Normalized_BATCH5-LR0.0005-Shrink0.98-StartAnneal2_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 19.80%
um_w2v2enlarge_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_NormSig-Web-W2V2-Large-En-Normalized_BATCH5-LR0.0005-Shrink0.98-StartAnneal2_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 24.44%
um_w2v2Sfrbase_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_LIG-FlowBBERT-Fr-Normalized_BATCH5-LR0.0005-Shrink0.98-StartAnneal2_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 23.11%
um_w2v2Sfrlarge_token_decoder_file='./END2END_Norm-Singals_Kheops-LSTM3L-H256_Drop0.5_FlowBERT-Fr-Normalized_BATCH5-LR0.0005-Shrink0.98-StartAnneal2_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 18.48%
um_w2v2Mfrbase_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_FlowBBERT-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 14.97%
um_w2v2Mfrlarge_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_FlowBERT-M-Fr-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 11.77%
um_xlsr53large_token_decoder_file='./END2END_Kheops-LSTM3L-H256_Drop0.5_XLSR53-ML-Normalized_BATCH5-LR0.000510204-Shrink0.98-StartAnneal1_BASIC-token_FlowBERT-TEST/checkpoint_best.pt' # WER 14.98%

warmup_opt=""
CUDA_VISIBLE_DEVICES=1 ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} --corpus-name ${CORPUS} \
	--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 \
	--decoder ${DECODER} --padded-reference \
	--save-dir ${SAVE_PATH} --patience 10 --no-epoch-checkpoints \
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
	--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum 1


