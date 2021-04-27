# Hybrid DNN-HMM ASR models 
Under construction...

Recipe scripts to train TDNN-F models in [Kaldi](https://github.com/kaldi-asr/kaldi) using hires and [Fairseq](https://github.com/pytorch/fairseq) wav2vec (2.0) features on the ETAPE-1,2 French corpus.

## Results
Results are reported in https://arxiv.org/pdf/2104.11462.pdf, Table 2.
<table>
  <thead>
    <tr>
      <th colspan="1">Language Model (LM)</th>
      <th colspan="2">LM ETAPE</th>
      <th colspan="2">LM ESTER-1,2+EPAC</th>
    </tr>
  </thead>
  <thead>
    <tr>
      <th>Features</th>
      <th>Dev</th>
      <th>Test</th>
      <th>Dev</th>
      <th>Test</th>
    </tr>
  </thead>
   
  <tbody>
   <tr>
    <td>hires MFCC</td>
    <td>39.28</td>
    <td>40.89</td>
    <td>35.60</td>
    <td>37.73</td>
   </tr>
   <tr>
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-M-large>W2V2-Fr-M-large</a></td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
   </tr>
   <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a></td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
   </tr>
   <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
   </tr>
  </tbody>
</table>
