
import os
import sys

from scipy import signal
from scipy.io import wavfile

import torch
import fairseq
import torch.nn.functional as F
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc

upsample = True
cuda = False
extract_features=True
model_size = 'large'
add_ext=True
file_ext = '.xlsr53-sv'
overwrite_arguments = {}

device = 'cuda:1' if cuda else 'cpu'
home='/home/getalp/dinarelm/'
media_path= home + '/work/data/MEDIA-Original/'
prefix_list= media_path + 'semantic_speech_aligned_corpus/media_run.lst'

f = open(prefix_list, encoding='utf-8')
lines = f.readlines()
f.close()
prefixes = [l.rstrip() for l in lines]

sv_starting_point='/home/getalp/dinarelm/work/data/Exchance/wav2vec/models/FlowBERT-7K_large_tuned-on-MEDIA_best-loss2.18.pt'
flowbert_path='/home/getalp/dinarelm/work/data/Exchance/wav2vec/models/XLSR53-large_sv-tuned_on-MEDIA_best-wer17.9.pt'

feat_norm = True if model_size == 'large' else False
if feat_norm and extract_features:
    print(' - Normalizing {} model features...'.format(model_size))
    sys.stdout.flush()
if upsample:
    print(' - Upsampling input signals...')
    sys.stdout.flush()

if extract_features:
    print(' - Loading model...')
    sys.stdout.flush()

    if sv_starting_point is not None:
        overwrite_arguments['model'] = {'w2v_path': sv_starting_point}
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([flowbert_path], overwrite_arguments)
    model = model[0]
    #state = torch.load(flowbert_path)
    #model.load_state_dict( state['model'] )
    model.eval()
    model = model.to(device)
else:
    model = None

for prefix in prefixes:
    if add_ext:
        wav_file = prefix + '.wav'
    else:
        wav_file = prefix
        prefix = prefix[:-4]

    print(' - Processing file {}'.format(wav_file.split('/')[-1]))
    sys.stdout.flush()

    sample_rate, samples = wavfile.read(wav_file,mmap = True)
    if upsample:
        samples = signal.resample(samples, samples.shape[0]*2)  # Up-sampling to 16kHz
        sample_rate = 2*sample_rate

    if extract_features:
        samples = torch.from_numpy(samples).float().squeeze() 
        samples = samples.to(device)
        if feat_norm:
            samples = F.layer_norm(samples, samples.shape)

        if isinstance(model, Wav2VecCtc):
            poker_face = torch.BoolTensor(1).fill_(False).to(samples.device)
            res = model.w2v_encoder.w2v_model.extract_features(source=samples.view(1,-1), padding_mask=poker_face, mask=False) 
            y = res[0]
        else:
            res = model(samples.view(1,-1), padding_mask=None, mask=False, features_only=True)
            y = res['x'] 

        torch.save(y.detach().to('cpu'), prefix + file_ext, _use_new_zipfile_serialization=False)
    elif upsample:
        wavfile.write(prefix + file_ext, sample_rate, samples)
    else:
        print('   * {}: samples {}, duration {} sec.'.format(wav_file.split('/')[-1], len(samples), len(samples)/sample_rate))
        feat_file = prefix + file_ext
        if os.path.isfile(feat_file):
            tt = torch.load(feat_file)
            print('   * {}: feature tensor size: {}'.format('/'.join(wav_file.split('/')[-3:]), tt.size()))



