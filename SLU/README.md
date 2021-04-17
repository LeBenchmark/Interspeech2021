# Spoken Language Understanding (SLU)

The SLU Benchmark used for the **LeBenchmark** is **MEDIA** (**link toward the data coming soon**).

The system used is based on Sequence-to-Sequence models and is coded for the [Fairseq library](https://github.com/pytorch/fairseq) (**code coming soon**).
The encoder is similar to the pyramidal LSTM-based encoder proposed in the [Listen, attend and spell paper](https://arxiv.org/abs/1508.01211), the only difference is that we compute the mean of two consecutive hidden states for reducing the output size between two layers, instead of concatenating them like in the original model.
The decoder is similar to the one used in our previous work published at [ICASSP 2020](http://www.marcodinarelli.it/publications/2020_ICASSP_EndToEndSLU.pdf). The differences in this case are that we use two attention mechanisms, one for attending the encoder hidden states, and the other for attending all the previous predictions, instead of using only the previous one like in the original decoder.

<table>
  <thead>
    <tr>
      <th></th>
      <th Model </th>
      <th Input Features </th>
      <th> DEV ER </th>
      <th> TEST ER </th>
    </tr>
  </thead>
  
  <tbody>
    <tr>
      <td> Kheops+Basic </td> <td> Spectrogram </td> <td> xx.xx </td> <td> xx.xx </td>
    </tr>
  </tbody>
 </table>
