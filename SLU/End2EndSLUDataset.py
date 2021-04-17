

import sys
import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset

# Use this to show warnings (anything else ?) at the command line
logger = logging.getLogger(__name__)

def collate_features(values, pad_idx, eos_idx=None, left_pad=False):
    """
    Convert a list of 2d tensors into a padded 3d tensor. 2d tensors are expected to be (length, dim)
    This function is intended to process speech input features, that is raw audio, spectrograms, etc.
    It thus does not make sense to pad or to add bos or eos symbols.
    """
    size = max(v.size(0) for v in values)
    dim = values[0].size(1)
    res = torch.zeros(len(values), size, dim).to(values[0])
    
    def copy_tensor(src, dst):
        assert dst.size(0) == src.size(0)
        assert dst.size(1) == src.size(1)
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i, size - v.size(0):,:] if left_pad else res[i, :v.size(0),:])
    return res

def collate_tokens_ex(values, pad_idx, bos_idx=None, eos_idx=None, left_pad=False, move_trail=False):
    """ Convert a list of 1d tensors into a padded 2d tensor.
        This is a generalization of the funcion in fairseq.data_utils which can either move eos to the beginning,
        or bos to the end (for backward decoders)
    """

    assert not ((bos_idx is not None) and (eos_idx is not None)), 'collate_tokens_ex: either bos index or eos index must be not None, got both None'

    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_trail:
            if bos_idx is not None:
                assert src[0] == bos_idx
                dst[-1] = bos_idx
                dst[:-1] = src[1:]
            else:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate(
            samples, pad_idx, bos_idx, eos_idx, left_pad_source=False, left_pad_target=False,
            input_feeding=True,
):

    if len(samples) == 0:
        return {}

    def merge_features(key, left_pad):
        return collate_features(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad,
        )
    
    def merge_tokens(key, left_pad, bos=None, eos=None, move_trail=False):
        return collate_tokens_ex(
            [s[key] for s in samples],
            pad_idx, bos_idx=bos, eos_idx=eos, left_pad=left_pad, move_trail=move_trail,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge_features('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = merge_tokens('target', left_pad=left_pad_target)
    target = target.index_select(0, sort_order).type(torch.LongTensor)
    tgt_lengths = torch.LongTensor([s['target'].size(0) for s in samples]).index_select(0, sort_order)
    ntokens = sum(len(s['target']) for s in samples)

    prev_output_tokens = None
    next_output_tokens = None
    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge_tokens(
            'target',
            left_pad=left_pad_target,
            bos=None,
            eos=eos_idx,
            move_trail=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order).type(torch.LongTensor)

        # This is created but not used for now...
        next_output_tokens = merge_tokens(
            'target',
            left_pad=left_pad_target,
            bos=bos_idx,
            eos=None,
            move_trail=True,
        )
        next_output_tokens = next_output_tokens.index_select(0, sort_order).type(torch.LongTensor)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'target_lengths' : tgt_lengths,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class End2EndSLUDataset(FairseqDataset):
    """
        A pair of torch.utils.data.Datasets. First containing feature tensors (e.g. wav signals, spectrograms, wav2vec features, etc.), second containing the desired output (e.g. characters, tokens, concepts, etc.)
        
        Args:
            src (torch.utils.data.Dataset): source dataset to wrap
            src_sizes (List[int]): source sentence lengths
            tgt (torch.utils.data.Dataset, optional): target dataset to wrap
            tgt_sizes (List[int], optional): target sentence lengths
            tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
            idx_structure (torch.utils.data.Dataset): indexes of sources for training preserving sources structures (e.g. dialogs, documents, etc.)
            left_pad_target (bool, optional): pad target tensors on the left side
                (default: False).
            max_source_positions (int, optional): max number of tokens in the
                source sentence (default: 1024).
            max_target_positions (int, optional): max number of tokens in the
                target sentence (default: 1024).
            shuffle (bool, optional): shuffle dataset elements before batching
                (default: True).
            input_feeding (bool, optional): create a shifted version of the targets
                to be passed into the model for teacher forcing (default: True).
            append_eos_to_target (bool, optional): if set, appends eos to end of
                target if it's absent (default: False).
            append_bos (bool, optional): if set, appends bos to the beginning of
                source/target sentence.
        """

    def __init__(
        self, src, src_sizes,
        tgt, tgt_sizes, tgt_dict,
        idx_structure,
        left_pad_target=False,
        max_source_positions=10000, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        append_eos_to_target=False,
        append_bos=False, eos=None
    ):

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.tgt_dict = tgt_dict

        self.idx_spk_batches = idx_structure
        self.idx_batches = []
        for s in idx_structure:
            self.idx_batches.append( [t[0] for t in s] )

        # Curriculum solution 1: sort all turns by length, without regard to the speaker
        lengths = [(i, t.size(0)) for i, t in enumerate(src)]
        sorted_structure = sorted(lengths, key=lambda tuple: tuple[1])
        self.curriculum_indices = [t[0] for t in sorted_structure]

        assert len(src) == len(self.curriculum_indices)

        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos() #(eos if eos is not None else tgt_dict.eos())

    def curriculum(self, value=False):
        self.shuffle = not value

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos()
            if self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos()
            if self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
            
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch with the following keys:
            
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                
                    - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                    - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                    - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                    
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                target sentence of shape `(bsz, tgt_len)`. Padding will appear
                on the left if *left_pad_target* is ``True``.
        """

        return collate(
            samples, pad_idx=self.tgt_dict.pad(), bos_idx=self.bos, eos_idx=self.eos,
            left_pad_source=False, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
            
        if self.shuffle:
            batch_shuffle_idx = np.random.permutation(len(self.idx_batches))
            indices = np.array([idx for sidx in batch_shuffle_idx for idx in self.idx_batches[sidx]])
        else:
            indices = np.array(self.curriculum_indices)
            return indices

        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)





































