
import os
import sys
import itertools
import argparse

import torch

blank_token = '__'
bos_token = 'SOS'
eos_token = 'EOS'
void_concept = 'null'
slu_start_concept_mark = '_SOC_'
slu_end_concept_mark = '_EOC_'

blank_idx = 0
sos_idx = -1
eos_idx = -1
soc_idx = -1
eoc_idx = -1
null_idx = -1

parser = argparse.ArgumentParser(description='Compute the error rate (i.e. edit distance)')
parser.add_argument('--ref', required=True, type=str, help='File containing reference sequences')
parser.add_argument('--hyp', required=True, type=str, help='File containing model hypotheses')
parser.add_argument('--show-progress-every', type=int, default=100, help='Show processing progress every given number of steps')
parser.add_argument('--clean-hyp', action='store_true', default=False, help='Remove duplicates from hypotheses (generated especially when CTC loss is used)')
parser.add_argument('--slu-scores', type=str, help='Use system scores for taking decisions, in case multiple concepts are found in the same chunk')
parser.add_argument('--slu-out', action='store_true', default=False, help='Keep only SLU labels')
parser.add_argument('--etape-slu-clean', action='store_true', default=False, help='Remove the bogus SLU annotation used in the ETAPE data')
args = parser.parse_args()

def tsr_edit_distance(ref, hyp):

    # These are 3, 3 and 4 in sclite if I remember well. However it does not change much the final error rate
    ins_weight = 1.0
    del_weight = 1.0
    sub_weight = 1.0
                    
    curr_x_size = ref.size(0)
    curr_y_size = hyp.size(0)
    tsr_ed_matrix = torch.FloatTensor(curr_x_size+1, curr_y_size+1).fill_(0)
    for i in range(1, curr_x_size+1):
        tsr_ed_matrix[i,0] = float(i)
    for i in range(1, curr_y_size+1):
        tsr_ed_matrix[0,i] = float(i)

    for ij in range(curr_x_size*curr_y_size):
        i = (ij // curr_y_size)+1
        j = (ij % curr_y_size)+1
                            
        tmp_weight = 0
        if ref[i-1] != hyp[j-1]:
            tmp_weight = sub_weight 
        tsr_ed_matrix[i,j] = min(tsr_ed_matrix[i-1,j] + del_weight, tsr_ed_matrix[i,j-1] + ins_weight, tsr_ed_matrix[i-1,j-1] + tmp_weight)

    # Back-tracking for error rate computation
    n_ins = 0
    n_del = 0
    n_sub = 0
    back_track_i = curr_x_size
    back_track_j = curr_y_size
    while back_track_i > 0 and back_track_j > 0:
        
        i = back_track_i
        j = back_track_j
        tmp_weight = 0
        if tsr_ed_matrix[i-1,j-1] != tsr_ed_matrix[i,j]:
            tmp_weight = sub_weight

        if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i,j-1]:
            if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i-1,j-1]:
                n_del += 1
                back_track_i -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                if tmp_weight > 0:
                    n_sub += 1
        else:
            if tsr_ed_matrix[i,j-1] < tsr_ed_matrix[i-1,j-1]:
                n_ins += 1
                back_track_j -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                if tmp_weight > 0:
                    n_sub += 1

    while back_track_i > 0:
        back_track_i -= 1
        n_del += 1

    while back_track_j > 0:
        back_track_j -= 1
        n_ins += 1
        
    return (n_ins, n_del, n_sub, curr_x_size)

def detect_concept_boundaries(concept_idx_lst, indeces):

    (blank_idx, start_concept_mark, end_concept_mark, null_idx, cleaning_flag, slu_flag) = indeces

    boundaries = []
    start = -1
    end = 0
    for idx, tok_idx in enumerate(concept_idx_lst):
        if tok_idx == start_concept_mark and start == -1:   # 1. Found a start mark, and nothing else happened => just mark the start
            start = idx
        elif tok_idx == start_concept_mark: # 2. Found a start, and another one was found before => a new chunk is starting, we auto-close the previous
            boundaries.append( (start, idx) )
            start = idx
        if tok_idx == end_concept_mark and start != -1: # 3. Ideal situation: found end, which closes a previous start => mark the end
            boundaries.append( (start, idx) )
            start = -1
            end = idx
        elif tok_idx == end_concept_mark:   # 4. found an end, but not preceded by a start => mark the end
            boundaries.append( (end, idx) )
            end = idx
    return boundaries


def normalize_semantics( concept_idx_lst, scores, indeces, concept_dict, inverse_dict ):

    if scores is not None:
        assert len(concept_idx_lst) == len(scores)

    greater_better_flag = False
    default_score = -999999.9 if greater_better_flag else 999999.9
    if greater_better_flag:
        def score_comp(curr, best):
            return curr > best
    else:
        def score_comp(curr, best):
            return curr < best

    (blank_idx, start_concept_mark, end_concept_mark, null_idx, cleaning_flag, slu_flag) = indeces
    if slu_flag:
        res_list = []
        boundaries = detect_concept_boundaries(concept_idx_lst, indeces)
        for b in boundaries:
            sem_found = False
            best_sem = ''
            best_score = default_score
            for sem_idx in range(b[1]-1,b[0],-1):
                sem = concept_idx_lst[sem_idx]
                score = scores[sem_idx] if scores is not None else default_score
                htoken = inverse_dict[sem]
                if sem != null_idx and sem != start_concept_mark and sem != end_concept_mark and htoken in concept_dict:
                    if not sem_found:
                        best_sem = sem
                        best_score = score
                        sem_found = True
                    else:
                        if score_comp(score, best_score):
                            best_score = score
                            best_sem = sem
            if sem_found:
                res_list.append(best_sem)
        return res_list
    elif null_idx != -1:
        res_list = []
        for i in concept_idx_lst:
            if i != null_idx:
                res_list.append(i)
        return res_list
    else:
        return concept_idx_lst

def unpad_sequence(sequence):
    
    start_idx = 0
    end_idx = len(sequence)
    if len(sequence) > 0:
        if sequence[0] == sos_idx:
            start_idx = 1
        if sequence[-1] == eos_idx:
            end_idx -= 1
        return sequence[start_idx:end_idx]
    return sequence

def compute_wer(hyp, scores, ref, indeces, concept_dict, inverse_dict):
    
    (blank_idx, start_concept_mark, end_concept_mark, null_idx, cleaning_flag, slu_flag) = indeces 
    output_idx = hyp

    new_scores = scores
    reverse_clean = True
    # The following statements are for cleaning the model output from CTC alignment mess, don't needed if another loss is used.
    remove_duplicates = cleaning_flag
    if remove_duplicates:
        if reverse_clean: 
            # 2. We remove duplicate items
            tmp_lst = [i.item() for i in output_idx]
            nodouble_lst = [k for k,g in itertools.groupby(tmp_lst)]

            new_scores = []
            if scores is not None:
                for ndl_el in nodouble_lst:
                    els_idx = [i for i,el in enumerate(tmp_lst) if el == ndl_el]
                    new_scores.append( min( [scores[idx] for idx in els_idx] ) )
                assert len(new_scores) == len(nodouble_lst) 

            output_hyp = torch.LongTensor( nodouble_lst )
            # 1. We remove blanks
            nonzero_idx = (output_hyp != blank_idx).nonzero()
            nonzero_idx = nonzero_idx.view(-1)
            output_hyp = output_hyp[nonzero_idx].type(ref.type())
            if scores is not None:
                new_scores = [new_scores[idx.item()] for idx in nonzero_idx]
            else:
                new_scores = None
        else:
            # 1. We remove blanks
            nonzero_idx = (output_idx != blank_idx).nonzero()
            nonzero_idx = nonzero_idx.view(-1)
            clean_hyp = output_idx[nonzero_idx].type(ref.type())
            # 2. We remove duplicate items
            tmp_lst = [i.item() for i in clean_hyp]
            nodouble_lst = [k for k,g in itertools.groupby(tmp_lst)]
            output_hyp = torch.LongTensor( nodouble_lst )
    else:
        output_hyp = output_idx
    nodouble_lst = normalize_semantics( [i.item() for i in output_hyp], new_scores, indeces, concept_dict, inverse_dict )
    clean_hyp = torch.LongTensor( unpad_sequence(nodouble_lst) )

    tmp_ref_lst = normalize_semantics( [i.item() for i in ref], None, indeces, concept_dict, inverse_dict )
    unpadded_list = unpad_sequence(tmp_ref_lst)
    ref = torch.IntTensor( unpadded_list ).view( len(unpadded_list), 1 )

    if ref.size(0) > 0:
        return tsr_edit_distance(ref, clean_hyp), [output_hyp, clean_hyp, ref], clean_hyp.size(0)
    else:
        return (clean_hyp.size(0), 0, 0, ref.size(0)), [output_hyp, clean_hyp, ref], clean_hyp.size(0)



dict = {}
inv_dict = {}
dict[blank_token] = len(dict)
inv_dict[dict[blank_token]] = blank_token
dict[bos_token] = len(dict)
inv_dict[dict[bos_token]] = bos_token
dict[eos_token] = len(dict)
inv_dict[dict[eos_token]] = eos_token
dict[void_concept] = len(dict)
inv_dict[dict[void_concept]] = void_concept

dict[slu_start_concept_mark] = len(dict)
inv_dict[dict[slu_start_concept_mark]] = slu_start_concept_mark
dict[slu_end_concept_mark] = len(dict)
inv_dict[dict[slu_end_concept_mark]] = slu_end_concept_mark

etape_clean_list = [void_concept, slu_start_concept_mark, slu_end_concept_mark]
blank_idx = dict[blank_token]
refs = []
ref_tokens = 0
sem_dict = {}
prev_token = ''
f = open(args.ref)
for line in f:
    line = line.rstrip("\r\n")
    tokens = line.split()
    ref_tokens += len(tokens)
    for t in tokens:
        if not t in dict:
            dict[t] = len(dict)
            inv_dict[dict[t]] = t
        if t == slu_end_concept_mark:
            if prev_token != '' and prev_token not in sem_dict:
                sem_dict[prev_token] = len(sem_dict)
                print(' - Adding {} to concept dictionary'.format(prev_token))
                sys.stdout.flush()
        prev_token = t
    refs.append( torch.LongTensor( [dict[t] for t in tokens if not t in etape_clean_list] ) ) if args.etape_slu_clean else refs.append( torch.LongTensor( [dict[t] for t in tokens] ) )
f.close()

print(' * Read {} lines, {} tokens, from file {}'.format(len(refs), ref_tokens, args.ref))
if args.slu_out:
    print('   - Concept dictionary size for SLU: {}'.format(len(sem_dict)))
print(' * Token dictionary size: {}'.format(len(dict)))
print('   * Inverse dictionary size: {}'.format(len(inv_dict)))
sys.stdout.flush()

def preprocess_hyp(hyp):
    res = []
    prev_t = ''
    for t in hyp:
        if prev_t != '' and t == slu_end_concept_mark and prev_t == blank_token: 
            res.append(void_concept)
        res.append(t)
        prev_t = t
    return res

hyps = []
hyp_tokens = 0
f = open(args.hyp)
for line in f:
    line = line.rstrip("\r\n")
    tokens = line.split()
    hyp_tokens += len(tokens)
    for t in tokens:
        if not t in dict:
            dict[t] = len(dict)
            inv_dict[dict[t]] = t 
    hyps.append( torch.LongTensor( [dict[t] for t in tokens if not t in etape_clean_list] ) ) if args.etape_slu_clean else hyps.append( torch.LongTensor( [dict[t] for t in tokens] ) )
f.close()

scores = []
nscores = 0
start_idx=1
if args.slu_scores:
    f = open(args.slu_scores)
    lines = f.readlines()
    scores = [l.rstrip().split()[start_idx:-1] for l in lines]
    nscores = sum( [len(s) for s in scores] )
    f.close()

print(' * Read {} lines, {} tokens, from file {}'.format(len(hyps), hyp_tokens, args.hyp))
if args.slu_scores:
    print('   * read {} lines, {} scores from file {}'.format(len(scores), nscores, args.slu_scores))
sys.stdout.flush()

if args.slu_scores:
    for idx in range(len(hyps)):
        if len(hyps[idx]) != len(scores[idx]):
            print(' - WARNING: length missmatch at sequence {}'.format(idx))
            sys.stdout.flush()

if len(refs) != len(hyps):
    sys.stderr.write(' FATAL ERROR, different number of references and hypotheses ({} VS. {})\n'.format(len(refs), len(hyps)))
    sys.exit(1)

total_er = [0,0,0]
total_toks = 0

blank_idx = dict[blank_token]
null_idx = dict[void_concept]
sos_idx = dict[bos_token]
eos_idx = dict[eos_token]
soc_idx = dict[slu_start_concept_mark]
eoc_idx = dict[slu_end_concept_mark]

clean_flag = args.clean_hyp
slu_flag = args.slu_out
scored_seqs = []
indeces = (blank_idx, soc_idx, eoc_idx, null_idx, clean_flag, slu_flag)
processed_hyps = 0
total_hyps = len(refs)
progress_format_str = '{:' + str(len(str(total_hyps))+1) + '} of {}\n'
for idx in range(len(refs)):

    sc = None
    if args.slu_scores:
        sc = scores[idx]

    (res, seq, nonzeros) = compute_wer(hyps[idx], sc, refs[idx], indeces, sem_dict, inv_dict)
    I, D, S, toks = res
    total_er[0] += I
    total_er[1] += D
    total_er[2] += S
    total_toks += toks

    (output_hyp, clean_hyp, ref) = seq
    hyp_txt = ' '.join( [inv_dict[idx.item()] for idx in clean_hyp] )
    ref_txt = ' '.join( [inv_dict[idx.item()] for idx in ref] )
    scored_seqs.append( (ref_txt, hyp_txt, sum(res[:3])) )

    processed_hyps += 1
    sys.stdout.write('.')
    if (processed_hyps % 10) == 0:
        sys.stdout.write('|')
    if (processed_hyps % args.show_progress_every) == 0:
        sys.stdout.write(progress_format_str.format(processed_hyps, total_hyps))
    sys.stdout.flush()
print('')

output_file = args.hyp + '.scored'
f = open(output_file, 'w')
for t in scored_seqs:
    f.write('REF: ' + t[0] + '\n')
    f.write('HYP: ' + t[1] + '\n')
    f.write(' - WER' + str(t[2]) + '\n\n')
f.close()

if total_toks == 0:
    total_toks = 1
    total_er = (0.33,0.33,0.34)
    print(' *** Got empty references! Setting error rate to 100%')

print(' ***')
print('{} VS. {}'.format(args.ref, args.hyp))
print('   * Total error rate: {:.3f}%'.format( float(sum(total_er))*100.0 / float(total_toks) ))
print('   * Total % corretc: {:.3f}%'.format( 100.0 - float(sum(total_er[1:]))*100.0 / float(total_toks) ))
print(' ***')
sys.stdout.flush()


























