# Hybrid DNN-HMM ASR models 
Under construction...

Recipe scripts to train TDNN-F models in [Kaldi](https://github.com/kaldi-asr/kaldi) using hires and [Fairseq](https://github.com/pytorch/fairseq) wav2vec (2.0) features on the ETAPE-1,2 French corpus.

## Data

ETAPE-1,2 French corpus: 30 hours of French radio and TV data

https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046/

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
      <th>Features (New name)</th>
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
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large>W2V2-Fr-M-large (W2V-Fr-3K-large) </a></td>
    <td>32.19</td>
    <td>33.87</td>
    <td>28.53</td>
    <td>30.77</td>
   </tr>
   <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a></td>
    <td>39.93</td>
    <td>42.30</td>
    <td>36.18</td>
    <td>38.75</td>
   </tr>
   <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>36.36</td>
    <td>38.19</td>
    <td>32.81</td>
    <td>35.17</td>
   </tr>
  </tbody>
</table>
