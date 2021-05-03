#!/usr/bin/env bash
# Recipe for training Kaldi ASR TDNN-F models on hires and wav2vec (2.0) features on ETAPE-1,2 
# Natalia Tomashenko 

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

nj=40 
decode_nj=10 
max_nj=512
mj=128

stage=0

. utils/parse_options.sh 

#if [ $stage -le 0 ]; then
  #TODO
  # download and prepare data: dev, test, train and lang for ETAPE-1,2
#fi 

if [ $stage -le 1 ]; then
  for dset in dev test train; do
	  dir=data/$dset
	  if [ ! -d $dir ]; then
		utils/copy_data_dir.sh data/${dset} $dir || exit 1
	  fi
	  [ ! -f $dir/spk2utt ] && echo "File $dir/spk2utt does not exist" && exit 1
	  nj=$(wc -l < $dir/spk2utt)
	  [ -z "$nj" ] && echo "Failed to get number of speakers for data set $dset" && exit 1
	  [ $nj -gt $max_nj ] && nj=$max_nj
	  steps/make_mfcc.sh \
		--nj $nj --cmd "$train_cmd --max-jobs-run $mj" \
		--write-utt2num-frames true \
		--mfcc-config conf/mfcc.conf $dir || exit 1
	  utils/fix_data_dir.sh $dir || exit 1
	  steps/compute_cmvn_stats.sh $dir || exit 1
  done
fi

if [ $stage -le 2 ]; then
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train data/lang exp/mono || exit 1
fi

if [ $stage -le 3 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali  || exit 1
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang exp/mono_ali exp/tri1  || exit 1
fi

if [ $stage -le 4 ]; then
  utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph_nosp || exit 1

  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset} || exit 1
  done
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1
fi

if [ $stage -le 6 ]; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph_nosp || exit 1
  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset} || exit 1
  done
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1

  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph || exit 1

  for dset in dev test; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3/graph data/${dset} exp/tri3/decode_${dset} || exit 1
  done
fi

if [ $stage -le 8 ]; then
  local/run_cleanup_segmentation.sh || exit 1
fi

# Train a TDNN-F model on hires features
if [ $stage -le 9 ]; then
  local/chain/run_tdnn_hires.sh || exit 1
fi

# Extract wav2vec (2.0) features
if [ $stage -le 10 ]; then
  # TODO download a model for wav2vec feature extraction
  # https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large          #French
  # https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt  #English
  # https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt   #Multi-lingual 
  mkdir -p data/models
  cd data/models 
  # wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt || exit 1  
  git lfs install || exit 1
  git clone https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large || exit 1
  ln -s wav2vec2-FR-M-large/checkpoint_best.pt  wav2vec2-FR-M-large.pt
  cd ..
  #model=data/models/xlsr_53_56k.pt
  model=data/models/wav2vec2-FR-M-large.pt
  for dset in dev test train_cleaned_sp; do
    utils/copy_data_dir.sh data/$dset data/${dset}_w2v
	$train_cmd --gpu 1 --num-threads 12 --time 24:00:00 data/${dset}_w2v/log/extract_wav2vec2.$dset.log \
	python local/extract_wav2vec.py $model data/${dset} data/${dset}_w2v || exit 1 
fi

# Train a TDNN-F model on wav2vec features using the same i-vectors and model topology
if [ $stage -le 11 ]; then
  local/chain/run_tdnn_w2v.sh || exit 1
fi

echo "$0: success."
exit 0
