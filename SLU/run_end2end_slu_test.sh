#!/bin/bash

# TO RUN ON THE LIG GRID WITH OAR UNCOMMENT THE FOLLOWING LINES
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/

# -----------------
FEATURES_TYPE='FlowBBERT' # Use 'spectro' or 'W2V'
FEATURES_LANG='Fr'      # Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
# -----------------
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
SERIALIZED_CORPUS=MEDIA.user+machine.${FEATURES_TYPE}-${FEATURES_LANG}-${NORMFLAG}.data
SUBTASK=concept

CRITERION='cross_entropy'

CHECKPOINT=checkpoints/checkpoint_best.pt
if [[ $# -ge 1 ]]; then
	CHECKPOINT=$1
fi
if [[ $# -eq 2 ]]; then
	SERIALIZED_CORPUS=$2
	if [[ `ls -l ${SERIALIZED_CORPUS}.* | wc -l` -eq 0 ]]; then
		echo "$0 ERROR: no serialized data found with prefix ${SERIALIZED_CORPUS}"; exit 1
	fi
fi

echo " * Using serialized data prefix: ${SERIALIZED_CORPUS}"
echo " ---"

GENERATE_OPTIONS="--beam 1 --iter-decode-max-iter 1 --max-len-a 1.0 --max-len-b 0 --prefix-size 0"

CHECKPOINT_DIR=`dirname ${CHECKPOINT}`
CUDA_VISIBLE_DEVICES=1 ${FAIRSEQ_PATH}/fairseq-generate ${DATA_PATH} \
	--path ${CHECKPOINT} ${GENERATE_OPTIONS} --max-sentences 1 --num-workers=0 \
	--task end2end_slu --criterion ${CRITERION} --padded-reference \
	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} --user-only \
	--gen-subset valid --results-path ${CHECKPOINT_DIR} \
