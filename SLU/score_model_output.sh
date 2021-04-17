#!/bin/bash

DECORE0_ID="_decore0"
source ${HOME}/work/tools/venv_python3.7.2_torch1.4${DECORE0_ID}/bin/activate
script_path=${HOME}/work/tools/fairseq_tools/end2end_slu/

model_output=$1

grep "^T\-" ${model_output} | cut -f 2- | perl -pe '{s/   / \# /g;}' > ${model_output}.ref # "s/   / \# \g;" is to keep into account spaces
grep "^H\-" ${model_output} | cut -f 3- | perl -pe '{s/   / \# /g;}' > ${model_output}.hyp
grep "^P\-" ${model_output} | cut -f 2- > ${model_output}.scores

# --clean-hyp removes blanks and duplicate tokens generated when training with CTC loss.
# Remove this option if you trained the model with another loss (e.g. cross entropy)
# --slu-out keeps only concepts from the raw output. Use this option if you want to score the model with Concept Error Rate (CER)
python ${script_path}/compute_error_rate.py --slu-out --clean-hyp --ref ${model_output}.ref --hyp ${model_output}.hyp --slu-scores ${model_output}.scores


