# Spoken Language Understanding (SLU)

The SLU Benchmark used for the **LeBenchmark** is **MEDIA** (**link toward the data coming soon**).

The system used is based on Sequence-to-Sequence models and is coded for the [Fairseq library](https://github.com/pytorch/fairseq) (**code coming soon**).
The encoder is similar to the pyramidal LSTM-based encoder proposed in the [Listen, attend and spell paper](https://arxiv.org/abs/1508.01211), the only difference is that we compute the mean of two consecutive hidden states for reducing the output size between two layers, instead of concatenating them like in the original model.
The decoder is similar to the one used in our previous work published at [ICASSP 2020](http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf). The differences in this case are that we use two attention mechanisms, one for attending the encoder hidden states, and the other for attending all the previous predictions, instead of using only the previous one like in the original decoder.

We use a similar training strategy as in [our previous work](http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf).
We train thus the encoder alone first, by putting a simple decoder (Basic) on top of it, that is a linear layer mapping the encoder hidden states into the output vocabulary size. The pre-trained encoders are used to pre-initialize parameters of models using a LSTM decoder (LSTM).
Results obtained with this strategy are summarized in the following table.
For more details please see the [Interspeech 2021 paper](?).

<table>
  <thead>
    <tr>
      <th colspan="4"> Token decoding (Word Error Rate)</th>
    </tr>  
    <tr>
      <th> Model </th>
      <th> Input Features </th>
      <th> DEV ER </th>
      <th> TEST ER </th>
    </tr>
  </thead>
  
  <tbody>
    <tr>
      <td> Kheops+Basic </td> <td> Spectrogram </td> <td> 36.25 </td> <td> 37.16 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2 En base </td> <td> 19.80 </td> <td> 21.78 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2 En large </td> <td> 24.44 </td> <td> 26.96 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2-S Fr base </td> <td> 23.11 </td> <td> 25.22 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2-S Fr large </td> <td> 18.48 </td> <td> 19.92 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2-M Fr base </td> <td> 14.97 </td> <td> 16.37 </td>
    </tr>
    <tr>
      <td> Kheops+Basic </td> <td> W2V2-M Fr large </td> <td> <b>11.77</b> </td> <td> <b>12.85</b> </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic </td> <td> XLSR53 large </td> <td> 14.98 </td> <td> 15.74 </td>
    </tr>
  </tbody>
  
  <thead>
    <tr>
      <th colspan="4"> Concept decoding (Concept Error Rate)</th>
    </tr>  
    <tr>
      <th> Model </th>
      <th> Input Features </th>
      <th> DEV ER </th>
      <th> TEST ER </th>
    </tr>
  </thead>
  
  <tbody>
    <tr>
      <td> Kheops+Basic </td> <td> Spectrogram </td> <td> 39.66 </td> <td> 40.76 </td>
    </tr>
    <tr>
      <td> Kheops+Basic +token </td> <td> Spectrogram </td> <td> 34.38 </td> <td> 34.74 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> Spectrogram </td> <td> 33.63 </td> <td> 34.76 </td>
    </tr>
    <thead>
      <tr>
        <th colspan="4"> </th>
      </tr>
    </thead>
    <tr>
      <td> Kheops+Basic +token </td> <td> W2V2 En base </td> <td> 26.79 </td> <td> 26.57 </td>
    </tr>
    <tr>
      <td> Kheops+LSTM +SLU </td> <td> W2V2 En base </td> <td> 26.31 </td> <td> 26.11 </td>
    </tr>
    
  </tbody>
  
 </table>
