#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import logging
import math
from itertools import groupby

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.globals import *
from fairseq.criterions import FairseqCriterion, register_criterion
from examples.speech_recognition.data.data_utils import encoder_padding_mask_to_lengths
from examples.speech_recognition.utils.wer_utils import Code, EditDistance, Token
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def arr_to_toks(arr):
    toks = []
    for a in arr:
        toks.append(Token(str(a), 0.0, 0.0))
    return toks


def compute_ctc_uer(logprobs, targets, input_lengths, target_lengths, blank_idx):
    """
        Computes utterance error rate for CTC outputs

        Args:
            logprobs: (Torch.tensor)  N, T1, D tensor of log probabilities out
                of the encoder
            targets: (Torch.tensor) N, T2 tensor of targets
            input_lengths: (Torch.tensor) lengths of inputs for each sample
            target_lengths: (Torch.tensor) lengths of targets for each sample
            blank_idx: (integer) id of blank symbol in target dictionary

        Returns:
            batch_errors: (float) errors in the batch
            batch_total: (float)  total number of valid samples in batch
    """
    batch_errors = 0.0
    batch_total = 0.0
    hyps = []
    refs = []
    for b in range(logprobs.shape[0]):
        predicted = logprobs[b][: input_lengths[b]].argmax(1).tolist()
        target = targets[b][: target_lengths[b]].tolist()

        # dedup predictions
        predicted = [p[0] for p in groupby(predicted)]

        # remove blanks
        nonblanks = []
        for p in predicted:
            if p != blank_idx:
                nonblanks.append(p)
        predicted = nonblanks

        #dedup predictions
        #predicted = [p[0] for p in groupby(predicted)]

        # compute the alignment based on EditDistance
        alignment = EditDistance(False).align(
            arr_to_toks(predicted), arr_to_toks(target)
        )

        # compute the number of errors
        # note that alignment.codes can also be used for computing
        # deletion, insersion and substitution error breakdowns in future
        for a in alignment.codes:
            if a != Code.match:
                batch_errors += 1
        batch_total += len(target)
        hyps.append( predicted )
        refs.append( target )

    return batch_errors, batch_total, hyps, refs


#@register_criterion("slubase_ctc_loss")
class SLUBaseCTCCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.blank_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.task = task
        self.args = args
        self.dictionary = task.target_dictionary
        self.end_concept = self.dictionary.add_symbol(slu_end_concept_mark)

        if args.slu_end2end:
            #self.aux_loss = CrossEntropyCriterion(task, False)
            self.aux_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)
        else:
            self.aux_loss = None
        self.loss_function = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--use-source-side-sample-size", action="store_true", default=False,
            help=(
                "when compute average loss, using number of source tokens "
                + "as denominator. "
                + "This argument will be no-op if sentence-avg is used."
            ),
        )

    def forward(self, model, sample, reduce=True, log_probs=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        sem_output = None
        if isinstance(net_output[1], dict) and 'c_outs' in net_output[1]:
            sem_output = net_output[1]['c_outs'] 
        bound_lprobs = model.get_normalized_probs(net_output, log_probs=log_probs) 

        # NOTE: I modified this since the original code was for training from the Encoder output, while I'm training with the Decoder output.
        #       Decoder output is (always ?) batch-first.

        batch_first = True
        (N, T, C) = bound_lprobs.size()  
        bound_input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(bound_lprobs.device)
        #input_lengths = encoder_padding_mask_to_lengths(
        #    net_output["encoder_padding_mask"], max_seq_len, bsz, device
        #)  # Decoder output doesn't contain 'encoder_padding_mask', see the NOTE above
        bound_target_lengths = sample["target_lengths"]
        bound_targets = sample["target"]    # B x T 

        if batch_first:
            # N T D -> T N D (F.ctc_loss expects this) 
            bound_lprobs = bound_lprobs.transpose(0, 1)

        pad_mask = sample["target"] != self.pad_idx
        bound_targets_flat = bound_targets.masked_select(pad_mask)

        #print('  *** Loss, bound_input_lengths shape: {}'.format(bound_input_lengths.size()))
        #sys.stdout.flush()
        
        loss = self.loss_function(
            bound_lprobs,
            bound_targets,
            bound_input_lengths,
            bound_target_lengths,
        )

        '''print(' *** CTC Loss, predicted and expected output shape:')
        print('   * Predicted: {}'.format(bound_lprobs.size()))
        print('   * Expected: {}'.format(bound_targets.size()))
        print(' *** CTC Loss.')
        sys.stdout.flush()'''

        if self.aux_loss is not None and sem_output is not None:
            #print(' *** Computing auxiliary loss...')
            #sys.stdout.flush()

            aux_net_output = (sem_output, None)
            sem_lprobs = model.get_normalized_probs(aux_net_output, log_probs=log_probs)
            sem_targets, sem_target_lengths = extract_concepts(bound_targets, self.dictionary.bos(), self.dictionary.eos(), self.dictionary.pad(), self.end_concept, move_trail=False)
            (aB, aT, aC) = sem_lprobs.size()

            #print('  *** Loss, sem_lprobs shape: {} x {} x {}'.format(aB, aT, aC))
            #sys.stdout.flush()

            sem_input_lengths = torch.full(size=(aB,), fill_value=aT, dtype=torch.long).to(sem_lprobs.device)
            if batch_first:
                sem_lprobs = sem_lprobs.transpose(0, 1)

            #print('  *** Loss, sem_input_lengths shape: {}'.format(sem_input_lengths.size()))
            #sys.stdout.flush()

            '''aux_loss = F.nll_loss(
                sem_lprobs.view(aB*aT, -1),
                sem_targets.view(aB*aT),
                ignore_index = self.aux_loss.padding_idx,
                reduction = 'sum' if reduce else 'none'
            )'''
            aux_loss = self.aux_loss(
                sem_lprobs,
                sem_targets,
                sem_input_lengths,
                sem_target_lengths
            )
            loss = loss + aux_loss

        bound_lprobs = bound_lprobs.transpose(0, 1)  # T N D -> N T D
        
        errors, total, hyps, refs = compute_ctc_uer(
            bound_lprobs.detach(), bound_targets.detach(), bound_input_lengths, bound_target_lengths, self.blank_idx
        )
        pad_count = 0
        if self.args.padded_reference:
            pad_count = N * 2

        batch_error_rate = float(errors) #/ (float(total) - pad_count)
        ber = torch.Tensor( [batch_error_rate] )

        #if self.args.sentence_avg:
        #    sample_size = sample["target"].size(0)
        #else:
            #if self.args.use_source_side_sample_size:
            #    sample_size = torch.sum(input_lengths).item()
            #else:
 
        sample_size = sample["ntokens"] -pad_count # if the reference is padded with bos and eos, do not account for them.
        logging_output = {
            "loss": utils.item( ber.data ) if reduce else ber.data, #utils.item(loss.data) if reduce else loss.data,
            "ctc": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample_size,
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "errors": errors,
            "total": total - pad_count,
            "predicted" : hyps,
            "target" : refs,
            "nframes": torch.sum(sample["net_input"]["src_lengths"]).item(),
        }

        #print(' - SLUCTCCriterion, loss computed, lprobs shape: {}'.format(lprobs.size()))
        #sys.stdout.flush() 

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        errors = sum(log.get("errors", 0) for log in logging_outputs)
        total = sum(log.get("total", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size, #/ math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "err": (errors * 100.0) / total,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item( sum(log.get("loss", 0) for log in logging_outputs) )
        ctc_sum = utils.item( sum(log.get("ctc", 0) for log in logging_outputs) )
        ntokens = utils.item( sum(log.get("ntokens", 0) for log in logging_outputs) )
        nsentences = utils.item( sum(log.get("nsentences", 0) for log in logging_outputs) )
        sample_size = utils.item( sum(log.get("sample_size", 0) for log in logging_outputs) )
        errors = utils.item( sum(log.get("errors", 0) for log in logging_outputs) )
        total = utils.item( sum(log.get("total", 0) for log in logging_outputs) )
        nframes = utils.item( sum(log.get("nframes", 0) for log in logging_outputs) )

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('ctc', ctc_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('err', (errors * 100.0) / total, total, round=4)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)
        metrics.log_scalar('nframes', nframes)
        metrics.log_scalar('sample_size', sample_size)


@register_criterion("slu_ctc_loss")
class SLUCTCCriterion(SLUBaseCTCCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.blank_idx = task.blank_idx
        self.eos_idx = task.label_vocab.eos()
        self.loss_function = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum',zero_infinity=True)

        '''if args.slu_end2end:
            print(' * SLUCTCCriterion, using an auxiliary loss for end2end SLU')
            sys.stdout.flush()
            #self.aux_loss = CrossEntropyCriterion(task, False)
            self.aux_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)
        else:
            self.aux_loss = None'''

        print(' - SLUCTCCriterion, initialized blank idx as {}'.format(self.blank_idx))
        sys.stdout.flush()

    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_criterion(cls, args, task):

        return SLUCTCCriterion(args, task)

