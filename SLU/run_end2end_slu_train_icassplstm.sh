#!/bin/bash

source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/

echo " ---"
echo "Using python: `which python`"
echo " ---"

# -----------------
FEATURES_TYPE='spectro'	# Use 'spectro', 'normspectro' or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' ... see below
FEATURES_SPEC=''		# Choose a meaningful infix to add into file names, you can leave it empty for spectrograms (*-spg)
FEATURES_EXTN='.20.0ms-spg'	# Feature file extension, e.g. '.20.0ms-spg', '.bert-3kb', '.bert-3kl', '.bert-3klt', ...
FEATURES_LANG='Fr'		# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
SUBTASK='concept'
CORPUS='media'
# -----------------

WORK_PATH=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
SERIALIZED_CORPUS=${WORK_PATH}/system_features/MEDIA.user+machine.${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}.data

# LSTMs in the encoder are bi-directional, thus the decoder size must be twice the size of the encoder
encoder_size=256
decoder_size=$((${encoder_size}*2))
ENC_FFN_DIM=$((${decoder_size}*2))
NCONV=1
NLSTM=3
ATT_HEADS=8
DLAYERS=2
DROP_RATIO=0.5
TRANS_DROP_RATIO=0.12
MAX_EPOCHS=100
LR=0.0005
LR_SHRINK=0.98
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
DECODER='icassp_lstm'
CRITERION='slu_ctc_loss' # Use 'cross_entropy' for Cross-Entropy loss, 'slu_ctc_loss' for CTC loss

if [[ $# -ge 1 ]]; then
	LR=$1
fi

ANNEAL=1
if [[ ${ANNEAL} -eq 1 ]]; then
        LR=`echo ${LR} ${LR_SHRINK} | awk '{print $1/$2}'`
fi
SAVE_PATH=${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}_Ziggurat121-Dec-${DECODER}-${SUBTASK}-H${encoder_size}-Drop${TRANS_DROP_RATIO}_BATCH${BATCH}_WU-Ep.2_forAVG5_TEST/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

wu_epochs=2
wu_updates=$((${wu_epochs}*5393))
warmup_opt="--warmup-updates ${wu_updates}"

#CUDA_VISIBLE_DEVICES=1
PYTHONPATH=${HOME}/work/tools/fairseq/ ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} \
	--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 \
	--decoder ${DECODER} --padded-reference \
	--save-dir ${SAVE_PATH} --patience 10 --no-epoch-checkpoints \
	--speech-conv ${NCONV} --num-features ${NUM_FEATURES} --speech-conv-size ${encoder_size} --drop-ratio ${DROP_RATIO} \
	--num-lstm-layers ${NLSTM} --speech-lstm-size ${encoder_size} --window-time ${WINTIME} --w2v-language ${FEATURES_LANG} \
	--encoder-normalize-before --encoder-layers 1 --encoder-attention-heads ${ATT_HEADS} --encoder-ffn-embed-dim ${ENC_FFN_DIM} \
	--decoder-normalize-before --attention-dropout ${TRANS_DROP_RATIO} --activation-dropout ${TRANS_DROP_RATIO} \
	--decoder-layers ${DLAYERS} --share-decoder-input-output-embed \
	--encoder-embed-dim ${encoder_size} --encoder-hidden-dim ${encoder_size} --decoder-embed-dim ${encoder_size} --decoder-hidden-dim ${encoder_size} \
	--dropout ${TRANS_DROP_RATIO} --decoder-dropout ${TRANS_DROP_RATIO} --decoder-attention-heads ${ATT_HEADS} \
	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} \
	--lr-scheduler fixed --lr ${LR} --force-anneal ${ANNEAL} --lr-shrink ${LR_SHRINK} ${warmup_opt} \
	--optimizer adam --clip-norm ${CLIP_NORM} --weight-decay ${WDECAY} \
	--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum 1 --keep-best-checkpoints 5

