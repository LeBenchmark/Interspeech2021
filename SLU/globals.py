
import sys
import torch

SOS_tag = 'SOS'
blank_token = "__"
pad_token = '_'
pad_char = pad_token
EOS_tag = 'EOS'
unk_token = '<unk>'
machine_semantic = 'MachineSemantic'
slu_start_concept_mark = '_SOC_'
slu_end_concept_mark = '_EOC_'
user_ID = 'User'
machine_ID = 'Machine'

LOSS_INIT_VALUE=999999.9
ER_INIT_VALUE=LOSS_INIT_VALUE

def extract_concepts(sequences, sos, eos, pad, eoc, move_trail=False):

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_trail: 
            assert src[-1] == eos
            dst[0] = eos
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    B, T = sequences.size() 
    concepts = []
    for i in range(B):
        concepts.append( torch.LongTensor( [sos] + [sequences[i,j-1] for j in range(T) if sequences[i,j] == eoc] + [eos] ).to(sequences) )
    lengths = [t.size(0) for t in concepts]
    max_len = max( lengths )
    res = torch.LongTensor(B, max_len).fill_(pad).to(sequences)
    for i in range(B): 
        copy_tensor(concepts[i], res[i,:concepts[i].size(0)])

    return res, torch.LongTensor( lengths ).to(sequences)

