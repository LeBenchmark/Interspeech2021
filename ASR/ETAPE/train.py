#!/usr/bin/env python3
import sys
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main

"""Recipe for training a sequence-to-sequence ASR system with CommonVoice.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard GRU and BeamSearch (no LM).

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all CommonVoice languages), and many
other possible variations.


Authors
 * Titouan Parcollet 2020
"""

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        ## Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


# Define custom data procedure
def dataio_prepare(hparams):

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
#    valid_data = valid_data.filtered_sorted(sort_key="duration")
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    datasets = [train_data, valid_data, test_data]

    # defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"]
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start_seg", "end_seg")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start_seg, end_seg):
#        info = torchaudio.info(wav)
        start = int(float(start_seg) * hparams["sample_rate"])
        stop = int(float(end_seg) * hparams["sample_rate"])
        speech_segment = {"file" : wav, "start" : start, "stop" : stop}
        sig = sb.dataio.dataio.read_audio(speech_segment)
        return sig

#        resampled = torchaudio.transforms.Resample(
#            info.sample_rate, hparams["sample_rate"],
#        )(sig)
#        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
#    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
#    run_on_main(
#        prepare_common_voice,
#        kwargs={
#            "data_folder": hparams["data_folder"],
#            "save_folder": hparams["save_folder"],
#            "train_tsv_file": hparams["train_tsv_file"],
#            "dev_tsv_file": hparams["dev_tsv_file"],
#            "test_tsv_file": hparams["test_tsv_file"],
#            "accented_letters": hparams["accented_letters"],
#            "language": hparams["language"],
#        },
#    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_set, tokenizer = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # Training
#    with torch.autograd.detect_anomaly():
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_set,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

