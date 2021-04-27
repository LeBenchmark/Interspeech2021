#!/usr/bin/env python3
# Extract wav2vec (2.0) features from wav files and segments
# Natalia Tomashenko

import torch
import fairseq
import argparse
import os
import numpy as np
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from torch import nn
import torch.nn.functional as F

from kaldiio import WriteHelper, ReadHelper

class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fname])
        self.normalize = ('normalize' in cfg['task']) and cfg['task']['normalize']
        model = model[0]
        model.eval()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            self.model.eval()
            if self.normalize:
                x = F.layer_norm(x, x.shape)
            return self.model.extract_features(x, padding_mask=None, mask=False)

class Prediction:

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            f, padding = self.model(x.unsqueeze(0))
           
        return f.squeeze(0).cpu().numpy()


def write_features(model, input, output):
    os.makedirs(output, exist_ok=True)
    with ReadHelper(f'ark:extract-segments scp:{input}/wav.scp {input}/segments ark:-|') as reader:
        with WriteHelper(f'ark,scp:{output}/feats.ark,{output}/feats.scp') as writer:
            for key, (sf, wav) in reader:
                wav = wav.astype(dtype=np.float32)
                feat = model(wav)
                feat = np.repeat(feat, 2, axis=0)
                writer(key, feat)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: extract_wav2vec.py <model> <input> <output>')
    parser.add_argument('model', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    print('Options:')
    print('  model: {}'.format(args.model))
    print('  input: {}'.format(args.input))
    print('  output: {}'.format(args.output))

    print('  Loading model...')
    model = Prediction(args.model)

    print('  Writing Features...')
    write_features(model, args.input, args.output)
    print('Done.')

