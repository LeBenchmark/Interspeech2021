# coding: utf8

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#sys.path.append( os.environ['RNNTAGGERPATH'] )
from fairseq.globals import *
#import utils_classes as Cl

# ---------- Decoders from LD-RNN tool  ----------
# This part is from my first system coded from scratch with pytorch, so be comprehensive if you see bullshits reading it :-)

class SimpleDecoder(nn.Module):

    def __init__(self, nn_params, input_size, direction):

        super(SimpleDecoder, self).__init__()

        # TMP FOR DEBUG
        self.debug_flag = False

        self.attention_heads = nn_params.attention_heads
        self.start_tag_idx = nn_params.start_tag_idx
        self.end_tag_idx = nn_params.end_tag_idx
        self.batch_size = nn_params.batch_size
        self.vocab_size = nn_params.word_vocab_size
        self.char_vocab_size = nn_params.char_vocab_size
        self.tagset_size = nn_params.tag_vocab_size
        self.hidden_dim = 2*nn_params.hidden_dim
        self.label_embed_dim = nn_params.label_embed_dim        # NEW
        self.char_embed_dim = nn_params.char_embed_dim
        self.char_hidden_dim = nn_params.char_hidden_dim
        self.label_context_size = nn_params.label_context_size
        self.lex_hidden_layers = nn_params.lex_hidden_layers
        self.lab_hidden_layers = nn_params.lab_hidden_layers
        
        # TMP FOR DEBUG
        #self.word_dict = nn_params.word_dict
        #self.label_dict = nn_params.label_dict
        #self.ix_to_sublabel = nn_params.ix_to_sublabel
        
        print(' - SimpleDecoder init:')
        print('   - start_tag_idx: {}'.format(self.start_tag_idx))
        print('   - end_tag_idx: {}'.format(self.end_tag_idx))
        print('   - batch_size: {}'.format(self.batch_size))
        print('   - vocab_size: {}'.format(self.vocab_size))
        print('   - char_vocab_size: {}'.format(self.char_vocab_size))
        print('   - tagset_size: {}'.format(self.tagset_size))
        print('   - hidden_dim: {}'.format(self.hidden_dim))
        print('   - label_embed_dim: {}'.format(self.label_embed_dim))
        print('   - char_embed_dim: {}'.format(self.char_embed_dim))
        print('   - char_hidden_dim: {}'.format(self.char_hidden_dim))
        print('   - label_context_size: {}'.format(self.label_context_size))
        print('   - lex_hidden_layers: {}'.format(self.lex_hidden_layers))
        print('   - lab_hidden_layers: {}'.format(self.lab_hidden_layers))
        print(' ----------')
        
        self.n_subparts = nn_params.n_subparts
        self.sl_batch_size = 1
        if self.n_subparts > 0:
            self.tag_to_subparts = nn_params.tag_to_subparts
        self.num_directions = 1
        self.CUDA = nn_params.CUDA
        self.TEST = 0
        self.TeachingSignal = True
        self.dtype = nn_params.dtype
        self.ltype = nn_params.ltype
        self.direction = direction
        self.output_length_factor = nn_params.output_length_factor
        if self.direction == 0 or self.direction == 1:
            self.output_length_factor = 1.0
        print(' *** SimpleDecoder, output-length-factor: {}'.format(self.output_length_factor))
        sys.stdout.flush()
    
        self.bw_label_embeddings = nn.Embedding(self.tagset_size, nn_params.label_embed_dim, sparse=False)
        self.emb_dropout_p = nn_params.embed_dropout        # NEW
        self.embed_dropout = nn.Dropout(p=nn_params.embed_dropout)

        self.attention_size = input_size    # TMP
        attention_size = input_size
        #self.hidden_dim = input_size + nn_params.label_context_size * nn_params.label_embed_dim + nn_params.attention_heads * attention_size
        #if self.n_subparts > 0:
        #    self.hidden_dim = self.hidden_dim + nn_params.sublabel_hidden_dim
        self.input_dim = input_size + nn_params.label_context_size * nn_params.label_embed_dim
        #if self.n_subparts > 0:
        #    self.input_dim = self.input_dim + nn_params.sublabel_hidden_dim
        
        self.BWInputNorm = nn.LayerNorm(self.input_dim)
        self.HiddenSizeMap = nn.Linear(self.input_dim, self.hidden_dim)
        if self.attention_heads > 0:
            print(' *** SimpleDecoder: using gated attention context')
            sys.stdout.flush()
            self.h_lin = nn.Linear(input_size, input_size)
            self.a_lin = nn.Linear(attention_size, attention_size)
            self.LexAttention = ga.GlobalAttention([self.hidden_dim, attention_size, attention_size], attention_size)
        #self.LexAttention = MultiHeadAttention(attention_size, self.hidden_dim, attention_size, nn_params.attention_heads, nn_params.attention_type, self.dtype) # NEW
        #self.SemAttention = AttentionModule(self.hidden_dim, self.hidden_dim, self.hidden_dim, nn_params.attention_heads, nn_params.attention_type, self.dtype) # NEW
        self.SLM = SubLabelModule(nn_params, input_size)
        
        self.RNNInputNorm = nn.LayerNorm(self.hidden_dim)
        #self.bw_RNN = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=False)
        self.bw_RNN = ContextualFeatureEncoder(self.hidden_dim, self.hidden_dim, self.batch_size, 1, False, nn_params.dtype, nn_params.contextual_encoder_type)
        #self.bw_RNN.flatten_parameters()
        
        self.MLPInputNorm = nn.LayerNorm(self.hidden_dim)
        self.BWOutputNorm = nn.LayerNorm(self.tagset_size)
        self.output_mlp = ReLU_MLP( [2,self.hidden_dim, self.hidden_dim] )
        output_dim = self.hidden_dim
        if self.n_subparts > 0:
            output_dim = output_dim + nn_params.sublabel_hidden_dim
        self.bw_hidden2tag = nn.Linear(output_dim, self.tagset_size)
        self.hid_dropout_p = nn_params.hidden_dropout        # NEW
        self.hidden_dropout = nn.Dropout(p=nn_params.hidden_dropout)
        
        #self.dir_hidden = self.bw_RNN.get_hidden_state()
    
    def init_hidden(self):
        #self.dir_hidden = torch.zeros(1, self.batch_size, self.hidden_dim).type(self.dtype) #VARIABLE
        self.bw_RNN.init_hidden()
        self.SLM.init_hidden()
    
    def resize_embeddings(self, nn_params):

        if nn_params.tag_vocab_size > self.tagset_size:
            old_embeddings = self.bw_label_embeddings
            self.bw_label_embeddings = nn.Embedding(nn_params.tag_vocab_size, nn_params.label_embed_dim, sparse=False)
            self.bw_label_embeddings.weight[:self.tagset_size,:] = old_embeddings.weight
            
            old_lin = self.bw_hidden2tag
            output_dim = self.hidden_dim
            if self.n_subparts > 0:
                output_dim = output_dim + nn_params.sublabel_hidden_dim
            self.bw_hidden2tag = nn.Linear(output_dim, nn_params.tag_vocab_size)
            self.bw_hidden2tag.weight[:self.tagset_size,:] = old_lin.weight
            
            old_norm = self.BWOutputNorm
            self.BWOutputNorm = nn.LayerNorm(nn_params.tag_vocab_size)
            self.BWOutputNorm.weight[:self.tagset_size] = old_norm.weight
            
            self.tagset_size = nn_params.tag_vocab_size


    def train_forward(self, input, bw_streams):
    
        dims = input[0].size()
        sequence_length = dims[0]
        batch_size = self.batch_size
        bw_label_streams = bw_streams[0]
        next_sublabels = bw_streams[1]
        
        indeces = decoding_indeces_(self.direction, sequence_length, self.output_length_factor)
        source_length = sequence_length
        sequence_length = len(indeces)
        gold_sequence_length = bw_label_streams[0].size(0)
        gold_to_hyp_length_factor = float(gold_sequence_length) / float(sequence_length)
        
        source_idxs = [int( i / self.output_length_factor ) for i in indeces]
        target_idxs = [int( i * gold_to_hyp_length_factor ) for i in indeces]
        
        # NEW: TEST IT!!!
        bin_size = 1
        if self.output_length_factor < 1.0:
            bin_size = int(1 / self.output_length_factor) + 1

        input_tsr = torch.cat( input, 2 )[source_idxs,:,:]
        local_input = [input_tsr]
        local_input.append( self.embed_dropout( self.bw_label_embeddings(bw_label_streams[0][target_idxs,:]) ) )

        # TMP FOR DEBUG
        if self.debug_flag:
            print('')
            print(' ************************************************')
            print(' * SimpleDecoder.train_forward -')
            print('')
            print(' *** indeces ({}): {}'.format(len(indeces), list(indeces)))
            print(' *** source_idxs ({}): {}'.format(len(source_idxs), source_idxs))
            print(' *** target_idxs ({}): {}'.format(len(target_idxs), target_idxs))
            print('*')
            print(' * Size of input: {}'.format( torch.cat(input, 2).size() ))
            print(' * Size of local_input: {}'.format( torch.cat(local_input, 2).size() ))
            print(' * Size of bw_label_streams: {}'.format(bw_label_streams[0].size()))
            print(' *')
            print(' * SimpleDecoder.train_forward, backward sublabels and labels:')
            for tgt_idx in target_idxs:
            #    print(' {}'.format([self.ix_to_sublabel[sl.item()] for sl in next_sublabels[tgt_idx,:,0]]))
                print(' -----')
                print('@{}, {}'.format(tgt_idx, self.label_dict.index2token(bw_label_streams[0][tgt_idx,0])))
            print('')
            print(' * SimpleDecoder.train_forward, len of local_input: {}'.format(len(local_input)))
            for debug_idx in range(len(local_input)):
                print('   * {}'.format(local_input[debug_idx].size()))
            print(' ---')
            #print(' * SimpleDecoder.train_forward, size of next_sublabels: {}'.format(next_sublabels.size()))
            print(' * SimpleDecoder.train_forward, size of bw_label_streams[0]: {}'.format(bw_label_streams[0].size()))
            print('')
        # END TMP FOR DEBUG

        bw_sublabels_rep = []
        if self.n_subparts > 0:
            bw_sublabels_rep = self.SLM( input_tsr, next_sublabels[target_idxs,:,:], 1 )

        # TMP FOR DEBUG
        if self.debug_flag:
            #print(' * SimpleDecoder.train_forward, size of bw_sublabels_rep: {}'.format(bw_sublabels_rep[0].size()))
            print(' ***********************************************************')
            sys.stdout.flush()

        #local_input = local_input + bw_sublabels_rep
        bw_total_input = self.BWInputNorm( torch.cat( local_input, 2 ) )
        
        #self.bw_RNN.flatten_parameters()
        #idxs = range(bw_total_input.size(0),-1,-1)
        rnn_input = self.RNNInputNorm( self.HiddenSizeMap( bw_total_input ) )
        bw_hidden_state, self.dir_hidden = self.bw_RNN( rnn_input )
        
        bw_mlp_input = rnn_input + self.hidden_dropout( bw_hidden_state )
        deep_reps = self.output_mlp( self.MLPInputNorm( bw_mlp_input ) )
        
        #bw_final_input = [bw_mlp_input + self.hidden_dropout( deep_reps )] + bw_sublabels_rep
        bw_final_input = torch.cat( [bw_mlp_input + self.hidden_dropout(deep_reps)] + bw_sublabels_rep, -1 )
        bw_scores = F.log_softmax( self.BWOutputNorm( self.bw_hidden2tag( bw_final_input ) ), dim=2 )
        
        return (bw_hidden_state, bw_scores)

    # NOTE: we assume "input" is a list of all inputs given to this layer, "bw_label_stream" is the stream of backward labels, so that accessing the i-th position of bw_label_stream when predicting label at position i, gives the label on the right of the current position.
    def fast_forward(self, input, bw_streams):
        
        vflag = (self.TEST == 1)
        if self.TeachingSignal and (not vflag):
            return self.train_forward(input, bw_streams)
        else:
            return self.test_forward(input, bw_streams)

    def test_forward(self, input, bw_streams):
        
        # NOTE: we assume the first element of input is the lexical-level representation computed by the encoder, that is its hidden state.
        #lex_rnn_out = input[0]
        vflag = (self.TEST == 1)
        dims = input[0].size()
        sequence_length = dims[0]
        batch_size = self.batch_size
        bw_label_streams = bw_streams[0]

        #print(' - SimpleDecoder.forward, input size: {}'.format(input[0].size()))
        #sys.stdout.flush()

        indeces = decoding_indeces_(self.direction, sequence_length, self.output_length_factor)
        source_length = sequence_length
        sequence_length = len(indeces)
        gold_sequence_length = bw_label_streams[0].size(0)
        gold_to_hyp_length_factor = float(gold_sequence_length) / float(sequence_length)
        
        embedding_mask = dropout_mask_dims( [1, batch_size, self.label_embed_dim], self.emb_dropout_p, self.dtype)
        hidden_layer_mask = dropout_mask_dims( [batch_size, self.hidden_dim], self.hid_dropout_p, self.dtype)
        if vflag:
            embedding_mask = torch.ones( [1, batch_size, self.label_embed_dim] ).type(self.dtype)
            hidden_layer_mask = torch.ones( [batch_size, self.hidden_dim] ).type(self.dtype)
        
        hidden_state = torch.zeros(sequence_length, batch_size, self.hidden_dim).type(self.dtype)    #VARIABLE
        scores = torch.zeros(sequence_length, batch_size, self.tagset_size).type(self.dtype) #VARIABLE
        start_idx = 0
        if self.direction == 1 or self.direction == 3:
            start_idx = -1
        next_labels = bw_label_streams[0][start_idx,:]
        prev_input = torch.cat( input, 2 )
        next_sublabels = bw_streams[1]  #VARIABLE
        
        # NEW: TEST IT!!!
        bin_size = 1
        if self.output_length_factor < 1.0:
            bin_size = int(1 / self.output_length_factor) + 1

        for i in indeces:
            source_idx = int( i / self.output_length_factor )
            bin_bound = min(source_length,source_idx+bin_size)    # NEW: TEXT IT!!!
            target_idx = int( i * gold_to_hyp_length_factor )

            if self.TeachingSignal and (not vflag):
                next_labels = bw_label_streams[0][target_idx,:]
                if self.n_subparts > 0:
                    next_sublabels = bw_streams[1][target_idx,:,:]  #VARIABLE   #GRAPHCHECKPOINT

            curr_lex_input = torch.sum( prev_input[source_idx:bin_bound,:,:], 0 )       # SOURCE INDEXING   ## This is ~different in 'train_forward'
            #curr_lex_input = prev_input[source_idx,:,:]         # TMP, SOURCE INDEXING ...
            bw_sublabels_rep = self.SLM( curr_lex_input, next_sublabels, 0 )
            bw_total_input_lst = [curr_lex_input.view(1, batch_size, -1)]   # SOURCE INDEXING   # NEW: TEST IT!!!
            bw_total_input_lst.append( self.embed_dropout( self.bw_label_embeddings( next_labels ).view(1, batch_size, -1) ) )
            
            if self.attention_heads > 0:
                #print(' xxx SimpleDecoder, applying attention: {}, {}'.format(hidden_state[i,:,:].size(), prev_input.size()))
                #sys.stdout.flush()
                c, alphas = self.LexAttention( hidden_state[i,:,:].clone().detach().view(batch_size, 1, -1), prev_input.transpose(0,1).contiguous().detach() )
                #bw_total_input_lst.append( c )
                # We gate-mix the original input and the attention vector
                g_lambda = F.sigmoid( self.h_lin( bw_total_input_lst[0] ) + self.a_lin(c) )
                bw_total_input_lst[0] = g_lambda * bw_total_input_lst[0] + (1.0 - g_lambda) * c

            bw_total_input = self.BWInputNorm( torch.cat( bw_total_input_lst, 2 ) )
            rnn_input = self.RNNInputNorm( self.hidden_dropout( self.HiddenSizeMap( bw_total_input ) ) )    # NEW: hidden_dropout !
            _, dec_hidden_state = self.bw_RNN( rnn_input )
            #hidden_state[i,:,:] = dec_hidden_state[0,:,:]

            bw_mlp_input = self.MLPInputNorm( rnn_input[0] + self.hidden_dropout( dec_hidden_state[0,:,:] ) )
            deep_reps = self.output_mlp( bw_mlp_input )
            
            final_dec_state = bw_mlp_input + self.hidden_dropout( deep_reps )
            hidden_state[i,:,:] = final_dec_state

            bw_final_input = torch.cat( [final_dec_state] + bw_sublabels_rep, -1 )
            scores[i,:,:] = F.log_softmax( self.BWOutputNorm( self.bw_hidden2tag( bw_final_input ) ), dim=1 )

            (max_scores, max_indeces) = torch.max(scores[i,:,:], 1)
            max_indeces = max_indeces.squeeze()
    
            if vflag:
                next_labels = max_indeces
                next_labels = next_labels.view(self.batch_size)
                max_indeces = max_indeces.unsqueeze(0)

                if self.n_subparts > 0:
                    next_sublabels = torch.LongTensor(self.tag_to_subparts[max_indeces].transpose(0,1)).type(self.ltype)   #VARIABLE   #GRAPHCHECKPOINT

        return (hidden_state, scores)

    def forward(self, input, bw_streams):

        return self.test_forward(input, bw_streams)

    def set_batch_size(self, val):
        self.batch_size = val
        if self.n_subparts > 0:
            self.sl_batch_size = val
        self.bw_RNN.set_batch_size( val )
        self.SLM.set_batch_size(val)

    def set_test_mode(self, val):
        self.TEST = val
        self.bw_RNN.set_test_mode( val )

    def set_teaching_signal_flag(self, val):

        self.TeachingSignal = val


class BidirectionalDecoder(nn.Module):

    def __init__(self, nn_params, input_size, direction):
    
        super(BidirectionalDecoder, self).__init__()
        
        # TMP FOR DEBUG
        self.debug_flag = False
        
        self.attention_heads = nn_params.attention_heads
        self.start_tag_idx = nn_params.start_tag_idx
        self.end_tag_idx = nn_params.end_tag_idx
        self.batch_size = nn_params.batch_size
        self.vocab_size = nn_params.word_vocab_size
        self.char_vocab_size = nn_params.char_vocab_size
        self.tagset_size = nn_params.tag_vocab_size
        self.hidden_dim = 2*nn_params.hidden_dim
        self.label_embed_dim = nn_params.label_embed_dim        # NEW
        self.char_embed_dim = nn_params.char_embed_dim
        self.char_hidden_dim = nn_params.char_hidden_dim
        self.label_context_size = nn_params.label_context_size
        self.lex_hidden_layers = nn_params.lex_hidden_layers
        self.lab_hidden_layers = nn_params.lab_hidden_layers
        self.n_subparts = nn_params.n_subparts
        self.sl_batch_size = 1
        if self.n_subparts > 0:
            self.tag_to_subparts = nn_params.tag_to_subparts
        self.num_directions = 1
        self.CUDA = nn_params.CUDA
        self.TEST = 0
        self.TeachingSignal = True
        self.dtype = nn_params.dtype
        self.ltype = nn_params.ltype
        self.direction = direction
        self.output_length_factor = nn_params.output_length_factor
        if self.direction == 0 or self.direction == 1:
            self.output_length_factor = 1.0
        
        # TMP FOR DEBUG
        #self.word_dict = nn_params.word_dict
        #self.label_dict = nn_params.label_dict
        #self.ix_to_sublabel = nn_params.ix_to_sublabel

        self.fw_label_embeddings = nn.Embedding(self.tagset_size, nn_params.label_embed_dim, sparse=False)
        self.emb_dropout_p = nn_params.embed_dropout        # NEW
        self.embed_dropout = nn.Dropout(p=nn_params.embed_dropout)
        
        attention_size = input_size
        sem_attention_size = self.hidden_dim
        self.input_dim = input_size + nn_params.label_context_size * nn_params.label_embed_dim
        
        self.FWInputNorm = nn.LayerNorm( self.input_dim )
        self.HiddenSizeMap = nn.Linear(self.input_dim, self.hidden_dim)
        if self.attention_heads > 0:
            self.h_lin = nn.Linear(attention_size, attention_size)
            self.a_lin = nn.Linear(attention_size, attention_size)
            self.LexAttention = ga.GlobalAttention([self.hidden_dim, attention_size, attention_size], attention_size)
            self.SemAttention = ga.GlobalAttention([self.hidden_dim, self.hidden_dim, self.hidden_dim], sem_attention_size)
        self.SLM = SubLabelModule(nn_params, input_size)

        self.RNNInputNorm = nn.LayerNorm( self.hidden_dim )
        self.fw_RNN = ContextualFeatureEncoder(self.hidden_dim, self.hidden_dim, self.batch_size, 1, False, nn_params.dtype, nn_params.contextual_encoder_type)
        self.hid_dropout_p = nn_params.hidden_dropout        # NEW
        self.hidden_dropout = nn.Dropout(p=nn_params.hidden_dropout)
        
        self.MLPInputNorm = nn.LayerNorm( self.hidden_dim )
        self.FWOutputNorm = nn.LayerNorm( self.tagset_size )
        self.output_mlp = ReLU_MLP( [2,self.hidden_dim, self.hidden_dim] )
        output_dim = self.hidden_dim
        if self.n_subparts > 0:
            output_dim = output_dim + nn_params.sublabel_hidden_dim
        output_dim = output_dim + nn_params.attention_heads * sem_attention_size
        self.hidden2tag = nn.Linear(output_dim, self.tagset_size)

        #self.dir_hidden = torch.zeros(1, self.batch_size, self.hidden_dim).type(self.dtype) #VARIABLE
    
    def init_hidden(self):
        #self.dir_hidden = torch.zeros(1, self.batch_size, self.hidden_dim).type(self.dtype) #VARIABLE
        self.fw_RNN.init_hidden()
        self.SLM.init_hidden()
    
    def resize_embeddings(self, nn_params):

        if nn_params.tag_vocab_size > self.tagset_size:
            old_embeddings = self.fw_label_embeddings
            self.fw_label_embeddings = nn.Embedding(nn_params.tag_vocab_size, nn_params.label_embed_dim, sparse=False)
            self.fw_label_embeddings.weight[:self.tagset_size,:] = old_embeddings.weight
            
            old_lin = self.hidden2tag
            output_dim = self.hidden_dim
            if self.n_subparts > 0:
                output_dim = output_dim + nn_params.sublabel_hidden_dim
            self.hidden2tag = nn.Linear(output_dim, nn_params.tag_vocab_size)
            self.hidden2tag.weight[:self.tagset_size,:] = old_lin.weight
            
            old_norm = self.FWOutputNorm
            self.FWOutputNorm = nn.LayerNorm(nn_params.tag_vocab_size)
            self.FWOutputNorm.weight[:self.tagset_size] = old_norm.weight
            
            self.tagset_size = nn_params.tag_vocab_size
    
    def train_forward(self, input, fw_streams, bw_states):

        dims = input[0].size()
        sequence_length = dims[0]
        batch_size = self.batch_size
        fw_label_streams = fw_streams[0]
        prev_sublabels = fw_streams[1]
        
        indeces = decoding_indeces_(self.direction, sequence_length, self.output_length_factor)
        source_length = sequence_length
        sequence_length = len(indeces)
        gold_sequence_length = fw_label_streams[0].size(0)
        gold_to_hyp_length_factor = float(gold_sequence_length) / float(sequence_length)
        
        source_idxs = [int( i / self.output_length_factor ) for i in indeces]
        target_idxs = [int( i * gold_to_hyp_length_factor ) for i in indeces]

        input_tsr = torch.cat( input, 2 )[source_idxs,:,:]
        local_input = [input_tsr]
        local_input.append( self.embed_dropout( self.fw_label_embeddings(fw_label_streams[0][target_idxs,:]) ) )

        # TMP FOR DEBUG
        if self.debug_flag:
            print('')
            print(' ************************************************')
            print(' * BidirectionalDecoder.train_forward -')
            print('')
            print(' *** indeces ({}): {}'.format(len(indeces), list(indeces)))
            print(' *** source_idxs ({}): {}'.format(len(source_idxs), source_idxs))
            print(' *** target_idxs ({}): {}'.format(len(target_idxs), target_idxs))
            print('*')
            print(' * Size of input: {}'.format( torch.cat(input, 2).size() ))
            print(' * Size of local_input: {}'.format( torch.cat(local_input, 2).size() ))
            print(' * Size of bw_label_streams: {}'.format(fw_label_streams[0].size()))
            print(' *')
            print(' * BidirectionalDecoder.train_forward, forward sublabels and labels:')
            for tgt_idx in target_idxs:
            #    print(' {}'.format([self.ix_to_sublabel[sl.item()] for sl in prev_sublabels[tgt_idx,:,0]]))
                print(' -----')
                print('@{}, {}'.format(tgt_idx, self.label_dict.index2token(fw_label_streams[0][tgt_idx,0])))
            print('')
            print(' * BidirectionalDecoder.train_forward, len of local_input: {}'.format(len(local_input)))
            for debug_idx in range(len(local_input)):
                print('   * {}'.format(local_input[debug_idx].size()))
            print(' ---')
            #print(' * BidirectionalDecoder.train_forward, size of prev_sublabels: {}'.format(prev_sublabels.size()))
            print(' * BidirectionalDecoder.train_forward, size of fw_label_streams[0]: {}'.format(fw_label_streams[0].size()))
            #print(' ***********************************************************')
            #print('')
        # END TMP FOR DEBUG

        fw_sublabels_rep = []
        if self.n_subparts > 0:
            fw_sublabels_rep = self.SLM( input_tsr, prev_sublabels[target_idxs,:,:], 1 )

        # TMP FOR DEBUG
        if self.debug_flag:
            #print(' * BidirectionalDecoder.train_forward, size of fw_sublabels_rep: {}'.format(fw_sublabels_rep[0].size()))
            print(' ***********************************************************')
            sys.stdout.flush()

        #local_input = local_input + fw_sublabels_rep
        fw_total_input = self.FWInputNorm( torch.cat( local_input, 2 ) )
        
        rnn_input = self.RNNInputNorm( self.HiddenSizeMap( fw_total_input ) )
        fw_hidden_state, self.dir_hidden = self.fw_RNN( rnn_input )
        
        fw_mlp_input = rnn_input + self.hidden_dropout( fw_hidden_state )
        deep_reps = self.output_mlp( self.MLPInputNorm( fw_mlp_input ) )
        
        fw_final_input = torch.cat( [fw_mlp_input + self.hidden_dropout( deep_reps + bw_states[0][indeces] )] + fw_sublabels_rep, -1 )
        fw_scores = F.log_softmax( self.FWOutputNorm( self.hidden2tag( fw_final_input ) ), dim=2 )
        
        return (fw_hidden_state, fw_scores)
    
    # NOTE: we assume "bw_states" contains backward hidden states and backward predictions, this and only this information, and in this order.
    # OBSOLETE: remove it !
    def fast_forward(self, input, fw_streams, bw_states):
        
        vflag = (self.TEST == 1)
        if self.TeachingSignal and (not vflag):
            #print(' * BidirectionalDecoder.train_forward...')
            #sys.stdout.flush()
            return self.train_forward(input, fw_streams, bw_states)
        else:
            #print(' * BidirectionalDecoder.test_forward...')
            #sys.stdout.flush()
            return self.test_forward(input, fw_streams, bw_states)


    # NOTE: we assume "bw_states" contains backward hidden states and backward predictions, this and only this information, and in this order.
    def test_forward(self, input, fw_streams, bw_states):
        
        # NOTE: we assume the first element of input is the lexical-level representation computed by the encoder, that is its hidden state.
        vflag = (self.TEST == 1)
        dims = input[0].size()
        sequence_length = dims[0]
        batch_size = self.batch_size
        fw_label_streams = fw_streams[0]
        
        target_length = bw_states[0].size(0)
        indeces = decoding_indeces_(self.direction, target_length, 1.0)     # We use the length of the output sequence predicted by a previous simple-decoder
        source_length = sequence_length
        sequence_length = len(indeces)
        gold_sequence_length = fw_label_streams[0].size(0)
        gold_to_hyp_length_factor = float(gold_sequence_length) / float(sequence_length)
        
        embedding_mask = dropout_mask_dims( [1, batch_size, self.label_embed_dim], self.emb_dropout_p, self.dtype)
        hidden_layer_mask = dropout_mask_dims( [batch_size, self.hidden_dim], self.hid_dropout_p, self.dtype)
        if vflag:
            embedding_mask = torch.ones( [1, batch_size, self.label_embed_dim] ).type(self.dtype)
            hidden_layer_mask = torch.ones( [batch_size, self.hidden_dim] ).type(self.dtype)

        fw_hidden_state = torch.zeros(sequence_length, batch_size, self.hidden_dim).type(self.dtype)    #VARIABLE
        fw_scores = torch.zeros(sequence_length, batch_size, self.tagset_size).type(self.dtype) #VARIABLE
        start_idx = 0
        if self.direction == 1 or self.direction == 3:
            start_idx = -1
        prev_labels = fw_label_streams[0][start_idx,:]
        prev_input = torch.cat( input, 2 )
        prev_sublabels = fw_streams[1]  #VARIABLE

        # NEW: TEST IT!!!
        bin_size = 1
        if self.output_length_factor < 1.0:
            bin_size = int(1 / self.output_length_factor) + 1

        self.fw_RNN.set_hidden_state( bw_states[0][0,:,:].view(1, batch_size, -1))
        for i in indeces:
            source_idx = int( i / self.output_length_factor )
            bin_bound = min(source_length,source_idx+bin_size)    # NEW: TEXT IT!!!
            target_idx = int( i * gold_to_hyp_length_factor )
            
            if self.TeachingSignal and (not vflag):
                prev_labels = fw_label_streams[0][target_idx,:]
                if self.n_subparts > 0:
                    prev_sublabels = fw_streams[1][target_idx,:,:]  #VARIABLE   #GRAPHCHECKPOINT

            curr_lex_input = torch.sum(prev_input[source_idx:bin_size,:,:],0)   ## This is ~different in 'train_forward'
            #curr_lex_input = prev_input[source_idx,:,:]
            fw_sublabels_rep = self.SLM( curr_lex_input, prev_sublabels, 0 )
            fw_total_input_lst = [curr_lex_input.view(1, batch_size, -1)]   # SOURCE INDEXING   # NEW: TEST IT!!!
            fw_total_input_lst.append( self.embed_dropout( self.fw_label_embeddings( prev_labels ).view(1, batch_size, -1) ) )
    
            if self.attention_heads > 0:
                c, alphas = self.LexAttention( fw_hidden_state[i,:,:].clone().view(batch_size, 1, -1), prev_input.transpose(0, 1).contiguous() )
                #fw_total_input_lst.append( c )
                g_lambda = F.sigmoid( self.h_lin( fw_total_input_lst[0] ) + self.a_lin(c) )
                fw_total_input_lst[0] = g_lambda * fw_total_input_lst[0] + (1.0 - g_lambda) * c
            
            fw_total_input = self.FWInputNorm( torch.cat( fw_total_input_lst, 2 ) )
            rnn_input = self.RNNInputNorm( self.hidden_dropout( self.HiddenSizeMap( fw_total_input ) ) )
            _, dec_hidden_state = self.fw_RNN( rnn_input )
            #fw_hidden_state[i,:,:] = dec_hidden_state[0,:,:]

            #mlp_input = fw_total_input[0] + hidden_layer_mask*( dec_hidden_state[0,:,:] )
            mlp_input = self.MLPInputNorm( rnn_input[0] + self.hidden_dropout( dec_hidden_state[0,:,:] ) )
            deep_reps = self.output_mlp( mlp_input )
            
            dec_final_state = mlp_input + self.hidden_dropout(deep_reps)
            fw_hidden_state[i,:,:] = dec_final_state
            atts = []
            if self.attention_heads > 0:
                sem_c, sem_alphas = self.SemAttention(dec_final_state.clone().view(batch_size, 1, -1), bw_states[0].transpose(0, 1).contiguous())
                atts = [sem_c.view(batch_size, -1)]
            
            #fw_final_input = torch.cat( [mlp_input + self.hidden_dropout(deep_reps) + bw_states[0][i,:,:]] + fw_sublabels_rep + atts, -1 )
            fw_final_input = torch.cat( [dec_final_state + bw_states[0][i,:,:]] + fw_sublabels_rep + atts, -1 )
            
            #fw_scores[i,:,:] = F.log_softmax( self.hidden2tag( fw_final_input + torch.sum( hidden_layer_mask*( torch.stack(fw_sem_atts) ) )), dim=1 )
            fw_scores[i,:,:] = F.log_softmax( self.FWOutputNorm( self.hidden2tag( fw_final_input ) ), dim=1 )

            (max_scores, max_indeces) = torch.max(fw_scores[i,:,:], 1)
            max_indeces = max_indeces.squeeze()
            
            if vflag:
                prev_labels = max_indeces
                prev_labels = prev_labels.view(self.batch_size)
                max_indeces = max_indeces.unsqueeze(0)

                if self.n_subparts > 0:
                    prev_sublabels = torch.LongTensor(self.tag_to_subparts[max_indeces].transpose(0,1)).type(self.ltype)   #VARIABLE   #GRAPHCHECKPOINT

        return (fw_hidden_state, fw_scores)

    def forward(self, input, fw_streams, bw_states):

        # TMP FOR DEBUG
        #self.train_forward(input, fw_streams, bw_states)
    
        return self.test_forward(input, fw_streams, bw_states)

    def set_batch_size(self, val):
        self.batch_size = val
        if self.n_subparts > 0:
            self.sl_batch_size = val
        self.fw_RNN.set_batch_size( val )
        self.SLM.set_batch_size(val)

    def set_test_mode(self, val):
        self.TEST = val
        self.fw_RNN.set_test_mode( val )

    def set_teaching_signal_flag(self, val):

        self.TeachingSignal = val

# ---------- Models for Speech decoding ----------

class Conv1dNormWrapper(nn.Module):
    '''
        class Conv1dNormWrapper
        
        Wrap a Conv1d class to be used in a nn.Sequential module, adding a layer normalization module.
    '''

    def __init__(self, input_size, output_size, kernel, stride_factor):

        super(Conv1dNormWrapper,self).__init__()

        self.conv = nn.Conv1d(input_size, output_size, kernel, stride=stride_factor)
        self.cNorm = nn.LayerNorm( output_size )

    def forward(self, input):

        return self.cNorm( self.conv( input ).permute(2,0,1) ).permute(1,2,0)

class LSTMWrapper(nn.Module):
    '''
        LSTMWrapper
        
        Wrap a LSTM layer to be used in a nn.Sequential module.
    '''

    def __init__(self, input_size, output_size, bidirFlag):

        super(LSTMWrapper,self).__init__()
        self.lstm = nn.LSTM(input_size, output_size, bidirectional=bidirFlag)

    def forward(self, input):

        output, _ = self.lstm( input )
        return output

class BasicEncoder(nn.Module):
    
    def __init__(self, params):
        
        super(BasicEncoder,self).__init__()
        #self.window_size = params.window_size

        # Parameter initialization
        # 1. Size of convolution layer
        self.input_size = params.num_features
        self.input_conv = self.input_size
        self.speech_conv_size = params.speech_conv_size
        
        # 2. Size of LSTM layer
        self.input_size_lstm = self.speech_conv_size
        self.hidden_size = params.speech_lstm_size
        
        # 3. Size of the output, that is of the linear layer
        self.output_size = params.output_size
        
        self.num_conv = params.speech_conv
        self.num_lstm_layers = params.num_lstm_layers
        self.conv_kernel = params.conv_kernel
        self.conv_kernel_width = params.conv_kernel_width
        self.conv_kernel_height = params.conv_kernel_height
        self.conv2d_dim = params.small_dim
        self.kernel_2d_hw_ratio = params.kernel_2d_hw_ratio
        self.stride_factor1 = params.conv_stride1
        self.stride_factor2 = params.conv_stride2

        # Layer initialization
        # 1. Convolutions
        conv_layers = []
        for i in range(self.num_conv):
            conv_stride = 1
            if i == self.num_conv-1:
                conv_stride = 2
            input_size = self.speech_conv_size
            if i == 0:
                input_size = self.input_conv
            conv_layers.append( ('Conv'+str(i+1), Conv1dNormWrapper(input_size, self.speech_conv_size, self.conv_kernel, conv_stride)) )
            conv_layers.append( ('Dropout'+str(i+1), nn.Dropout(p=params.drop_ratio)) )
            #conv_layers.append( ('ConvNorm'+str(i+1), nn.BatchNorm1d( self.speech_conv_size )) )
        self.convolutions = nn.Sequential( OrderedDict(conv_layers) )
        
        '''#self.conv1 = nn.Conv2d(self.input_conv,self.speech_conv_size, (self.conv_kernel_width, self.conv_kernel_height), stride=(self.stride_factor1, self.stride_factor1))
        self.conv1 = nn.Conv1d(self.input_conv,self.speech_conv_size, self.conv_kernel, stride=self.stride_factor1)
        #self.conv2 = nn.Conv1d(self.speech_conv_size,self.speech_conv_size,self.conv_kernel,stride=self.stride_factor2)'''
        #self.CONV_norm = nn.LayerNorm( self.speech_conv_size )
        
        # 2. Recurrent layers
        recurrent_layers = []
        for i in range(self.num_lstm_layers):
            input_size = 2*self.hidden_size
            if i == 0:
                input_size = self.input_size_lstm
            recurrent_layers.append( ('LSTM'+str(i+1), LSTMWrapper(input_size, self.hidden_size, True)) )
            recurrent_layers.append( ('ConvNorm'+str(i+1), nn.LayerNorm( 2*self.hidden_size )) )
            recurrent_layers.append( ('Dropout'+str(i+1), nn.Dropout(p=params.drop_ratio)) )
        self.rnns = nn.Sequential( OrderedDict(recurrent_layers) )

        #self.h_dropout = nn.Dropout(p=params.drop_ratio)
        #self.LSTM_norm = nn.LayerNorm(self.hidden_size*2)
        #self.rnns = nn.LSTM(self.input_size_lstm,self.hidden_size,num_layers = self.num_lstm_layers,bidirectional=True)
        
        #Linear Layer
        self.linear_layer = nn.Linear(2*self.hidden_size, self.output_size)
        
        #small_dim = int( math.sqrt(seq_len / hw_ratio) + 0.5 )
        #x_pad = torch.randn(num_features, batch_size, small_dim * hw_ratio * small_dim - seq_len)
        #x_padded = torch.cat( [x, x_pad], 2 )
        #x_conv = x_padded.view(num_features, batch_size, hw_ratio*small_dim, small_dim)

        '''
        print(' *** Initializing BasicEncoder:')
        print('   * Input size: {}'.format(params.num_features))
        print('   * Output size: {}'.format(params.output_size))
        print('   * Convolution size: {}'.format(params.speech_conv_size))
        print('   * Hidden size: {}'.format(params.speech_lstm_size))
        print('   -')
        print('   * Stride factor 1: {}'.format(params.conv_stride1))
        print('   * Stride factor 2: {}'.format(params.conv_stride2))
        print('   * Num. LSTM layers: {}'.format(params.num_lstm_layers))
        print(' ***')
        '''


    def forward(self, x):
        # Input has shape (sequence_length, batch_size, num. of channels), that is (L, N, C), convolution needs it to be (N, C, L)
        
        # 1. For Conv2d
        #(L, N, C) = x.size()
        #small_dim = int( math.sqrt(float(L) / float(self.kernel_2d_hw_ratio)) )
        #out = self.conv1( x.permute(1, 2, 0).view(N, C, small_dim * self.kernel_2d_hw_ratio, small_dim) )
        #out = self.h_dropout( out.view(N, self.speech_conv_size, -1).permute(2,0,1) )
        # ---------------------
        
        '''# 2. For Conv1d
        out = self.conv1( x.permute(1, 2, 0) )
        out = self.h_dropout( out.permute(2,0,1) )
        # ---------------------
        
        #out = self.conv2(x)

        output, _ = self.rnns( self.conv_output_norm( out ) )
        output = self.h_dropout(output)
        
        output = self.linear_layer( self.LSTM_norm(output) )
        #output = self.log_softmax(output)'''
        
        # New forward code with generic layer structures
        out = self.convolutions( x.permute(1, 2, 0) )
        #out = self.rnns( self.CONV_norm( out.permute(2,0,1) ) )
        #output = self.linear_layer( self.LSTM_norm( out ) )
        out = self.rnns( out.permute(2, 0, 1) )
        output = self.linear_layer( out )
        
        return (output, output, out)


class BasicSpeechEncoder(nn.Module):

    def __init__(self, params, nn_params):

        super(BasicSpeechEncoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]

        self.encoder = BasicEncoder(params)
        self.log_softmax = nn.LogSoftmax(dim = 2)

    def get_fw_parameters(self):
        return self.parameters()

    def get_bw_parameters(self):
        return self.get_fw_parameters()

    def forward(self, x, next_labels, prev_labels):

        (representations, reps, hidden_states) = self.encoder( x )
        scores = self.log_softmax( representations )

        return (scores, scores, hidden_states)     # SWITCH TO THIS FOR RICH-REPRESENTATION ARCHITECTURE
        #return (scores, representations)

    def set_test_mode(self, val):
    
        return

    def set_teaching_signal_flag(self, val):

        return

    def set_speaker_val(self, val):

        self.speaker_val = val

    def pad_input(self, input, val):

        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        #SpkID = torch.ones_like(input)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
            #SpkID[:,i,:] = SpkID[:,i,:] * self.speaker_val[i] * 0.002
        #return torch.cat( [padder, input + SpkID, padder], 0 )
        return torch.cat( [padder, input, padder], 0 )

class BasicSpeechSeqEncoder(nn.Module):
    
    def __init__(self, params, nn_params):
        
        super(BasicSpeechSeqEncoder,self).__init__()
            
        self.speaker_val = [globals.user_speaker_val]
            
        self.encoder = BasicEncoder(params)
        self.seq_encoder = SimpleDecoder(nn_params, 2*params.speech_lstm_size, 0)

    def get_fw_parameters(self):
        
        return self.parameters()

    def get_bw_parameters(self):

        return self.get_fw_parameters()

    def forward(self, x, next_labels, prev_labels):
        
        (sequence_length, batch_size, num_features) = x.size()
        self.seq_encoder.set_batch_size( batch_size )
                                
        (representations, reps, hidden_states) = self.encoder(x)

        (prev_sublabels, next_sublabels) = (torch.LongTensor([0]),torch.LongTensor([0]))
        fw_streams = (prev_labels, prev_sublabels)
                                                                
        self.seq_encoder.init_hidden()
        (fw_hidden_state, fw_scores) = self.seq_encoder([hidden_states], fw_streams)   # SWITCH TO THIS FOR RICH-REPRESENTATION ARCHITECTURE

        return (fw_scores, fw_scores, fw_hidden_state)
    
    def set_test_mode(self, val):
        
        self.seq_encoder.set_test_mode( val )
       
    def set_teaching_signal_flag(self, val):
        
        self.seq_encoder.set_teaching_signal_flag( val )
        
    def load_encoder(self, bsencoder):
        
        self.encoder.load_state_dict( bsencoder.encoder.state_dict() )
        
    def set_speaker_val(self, val):
        
        self.speaker_val = val
        
    def pad_input(self, input, val):
        
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.cuda.FloatTensor(1, batch_size, num_features)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

class BasicSpeechBiseqEncoder(nn.Module):
    
    def __init__(self, params, nn_params):
        
        super(BasicSpeechBiseqEncoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]
        
        self.encoder = BasicEncoder(params)
        #self.seq_encoder = SimpleDecoder(nn_params, params.output_size, 0)
        #self.seq_encoder = SimpleDecoder(nn_params, params.output_size, 2)   # NEW: TEST IT!!!
        self.bw_seq_encoder = SimpleDecoder(nn_params, 2*params.speech_lstm_size, 1)  # SWITCH TO THIS FOR RICH-REPRESENTATION ARCHITECTURE
        #self.log_softmax = nn.LogSoftmax(dim = 2)
        self.fw_seq_encoder = BidirectionalDecoder(nn_params, 2*params.speech_lstm_size, 0)

    def get_fw_parameters(self):

        return list(filter(lambda p: p.requires_grad, self.encoder.parameters())) + list(filter(lambda p: p.requires_grad, self.fw_seq_encoder.parameters()))

    def get_bw_parameters(self):

        return list(filter(lambda p: p.requires_grad, self.encoder.parameters())) + list(filter(lambda p: p.requires_grad, self.bw_seq_encoder.parameters()))
    
    def forward(self, x, next_labels, prev_labels):
        
        (sequence_length, batch_size, num_features) = x.size()
        self.fw_seq_encoder.set_batch_size( batch_size )
        self.bw_seq_encoder.set_batch_size( batch_size )
        
        (representations, reps, hidden_states) = self.encoder(x)

        (prev_sublabels, next_sublabels) = (torch.LongTensor([0]),torch.LongTensor([0]))
        fw_streams = (prev_labels, prev_sublabels)
        bw_streams = (next_labels, next_sublabels)
        
        self.bw_seq_encoder.init_hidden()
        self.fw_seq_encoder.init_hidden()
        #(fw_hidden_state, fw_scores) = self.seq_encoder([representations], fw_streams)
        (bw_hidden_state, bw_scores) = self.bw_seq_encoder([hidden_states], bw_streams)   # SWITCH TO THIS FOR RICH-REPRESENTATION ARCHITECTURE
        (fw_hidden_state, fw_scores) = self.fw_seq_encoder([hidden_states], fw_streams, [bw_hidden_state, bw_scores])
        global_scores = 0.5 * (fw_scores + bw_scores)

        return (fw_scores, bw_scores, fw_hidden_state)

    def set_test_mode(self, val):

        self.bw_seq_encoder.set_test_mode( val )
        self.fw_seq_encoder.set_test_mode( val )

    def set_teaching_signal_flag(self, val):

        self.bw_seq_encoder.set_teaching_signal_flag( val )
        self.fw_seq_encoder.set_teaching_signal_flag( val )

    def load_encoder(self, bsencoder):

        self.encoder.load_state_dict( bsencoder.encoder.state_dict() )

    def set_speaker_val(self, val):

        self.speaker_val = val

    def pad_input(self, input, val):
    
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

        #self.speaker_val = val
        #(sequence_length, batch_size, num_features) = input.size()
        #padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        #SpkID = torch.ones_like(input)
        #for i in range( batch_size ):
        #    padder[:,i,:] = self.speaker_val[i]
        #    SpkID[:,i,:] = SpkID[:,i,:] * self.speaker_val[i] * 0.002
        #return torch.cat( [padder, input + SpkID, padder], 0 )

class MLSpeechEncoder(nn.Module):

    def __init__(self, ch_params, tk_params, nn_params):

        super(MLSpeechEncoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]

        self.char_encoder = BasicSpeechEncoder(ch_params, nn_params)
        self.token_encoder = BasicSpeechEncoder(tk_params, nn_params)

    def get_fw_parameters(self):

        return self.parameters()

    def get_bw_parameters(self):

        return self.get_fw_parameters()

    def forward(self, x, next_labels, prev_labels):

        (ch_scores, ch_sc, ch_reps) = self.char_encoder(x, next_labels, prev_labels)
        (tk_scores, tk_sc, tk_reps) = self.token_encoder(ch_reps, next_labels, prev_labels)

        return (tk_scores, tk_scores, tk_reps)

    def load_char_encoder(self, char_encoder):

        self.char_encoder.encoder.load_state_dict( char_encoder.encoder.state_dict() )
        #for param in self.char_encoder.encoder.parameters():
        #    param.requires_grad = False
        
    def freeze_char_encoder(self):
    
        for param in self.char_encoder.parameters():
            param.requires_grad = False

    def unfreeze_char_encoder(self):
    
        for param in self.char_encoder.parameters():
            param.requires_grad = True

    def load_token_encoder(self, token_encoder):

        self.token_encoder.encoder.rnns.load_state_dict( token_encoder.encoder.rnns.state_dict() )

    def set_test_mode(self, val):
    
        return

    def set_teaching_signal_flag(self, val):
        
        return

    def set_speaker_val(self, val):

        self.speaker_val = val

    def pad_input(self, input, val):
    
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

class MLSpeechSeqEncoder(nn.Module):
    
    def __init__(self, ch_params, tk_params, nn_params):
        
        super(MLSpeechSeqEncoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]
        
        self.char_encoder = BasicSpeechEncoder(ch_params, nn_params)
        self.token_encoder = BasicSpeechSeqEncoder(tk_params, nn_params)

    def get_fw_parameters(self):

        return self.char_encoder.get_fw_parameters() + self.token_encoder.get_fw_parameters()

    def get_bw_parameters(self):

        return self.char_encoder.get_bw_parameters() + self.token_encoder.get_bw_parameters()

    def forward(self, x, next_labels, prev_labels):
        
        (ch_scores, ch_sc, ch_reps) = self.char_encoder(x, next_labels, prev_labels)
        (fw_tk_scores, bw_tk_scores, tk_reps) = self.token_encoder(ch_reps, next_labels, prev_labels)
        
        return (fw_tk_scores, bw_tk_scores, tk_reps)
    
    def load_char_encoder(self, char_encoder):
        
        self.char_encoder.encoder.load_state_dict( char_encoder.encoder.state_dict() )
        #for param in self.char_encoder.encoder.parameters():
        #    param.requires_grad = False
    
    def freeze_char_encoder(self):
        
        for param in self.char_encoder.parameters():
            param.requires_grad = False

    def unfreeze_char_encoder(self):
    
        for param in self.char_encoder.parameters():
            param.requires_grad = True

    def load_token_encoder(self, token_encoder):
        
        self.token_encoder.encoder.rnns.load_state_dict( token_encoder.encoder.rnns.state_dict() )
        self.token_encoder.bw_seq_encoder.load_state_dict( token_encoder.bw_seq_encoder.state_dict() )
        self.token_encoder.fw_seq_encoder.load_state_dict( token_encoder.fw_seq_encoder.state_dict() )

    def load_ml_encoder(self, ml_encoder):
        
        self.char_encoder.load_state_dict( ml_encoder.char_encoder.state_dict() )
        #print(' -- MLSpeechSeqEncoder: freezing char-encoder parameters...')
        #for param in self.char_encoder.parameters():
        #    param.requires_grad = False
        self.token_encoder.encoder.load_state_dict( ml_encoder.token_encoder.encoder.state_dict() )
        #print(' -- MLSpeechSeqEncoder: freezing token-encoder (encoder only) parameters...')
        #sys.stdout.flush()
        #for param in self.token_encoder.encoder.parameters():
        #    param.requires_grad = False
    
    def load_ml_seq_decoder(self, ml_encoder):
        
        self.char_encoder.load_state_dict( ml_encoder.char_encoder.state_dict() )
        self.token_encoder.load_state_dict( ml_encoder.token_encoder.state_dict() )

    def set_test_mode(self, val):
    
        self.token_encoder.set_test_mode( val )

    def set_teaching_signal_flag(self, val):
        
        self.token_encoder.set_teaching_signal_flag( val )

    def set_speaker_val(self, val):

        self.speaker_val = val

    def pad_input(self, input, val):
    
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

# ---------- Models for End-to-end SLU ----------

class SLUSimpleDecoder(nn.Module):

    def __init__(self, ch_params, tk_params, nn_params):
        
        super(SLUSimpleDecoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]

        tmp = nn_params.tag_vocab_size
        nn_params.tag_vocab_size = nn_params.sd_tag_vocab_size
        decoder_output_size = 0
        if nn_params.train_char_decoder or nn_params.load_char_decoder:
            print(' -- SLUSimpleDecoder: using character speech decoder')
            sys.stdout.flush()
            self.speech_decoder = BasicSpeechSeqEncoder(ch_params, nn_params)
            decoder_output_size = nn_params.hidden_dim
        elif nn_params.train_token_decoder or nn_params.load_token_decoder:
            print(' -- SLUSimpleDecoder: using token speech decoder')
            sys.stdout.flush()
            self.speech_decoder = BasicSpeechSeqEncoder(tk_params, nn_params)
            decoder_output_size = nn_params.hidden_dim
        elif nn_params.train_ml_decoder or nn_params.load_ml_decoder:
            print(' -- SLUSimpleDecoder: using 2-stage token speech decoder')
            sys.stdout.flush()
            self.speech_decoder = MLSpeechSeqEncoder(ch_params, tk_params, nn_params)
            decoder_output_size = nn_params.hidden_dim

        nn_params.tag_vocab_size = tmp
        nn_params.label_embed_dim = 2 * nn_params.label_embed_dim
        nn_params.hidden_dim = 2 * nn_params.hidden_dim
        self.slu_decoder = SimpleDecoder(nn_params, decoder_output_size, 0)

    def get_fw_parameters(self):

        return self.speech_decoder.get_fw_parameters() + list(filter(lambda p: p.requires_grad, self.slu_decoder.parameters()))

    def get_bw_parameters(self):

        return self.speech_decoder.get_bw_parameters() + list(filter(lambda p: p.requires_grad, self.slu_decoder.parameters()))

    def forward(self, input, bw_label_streams, fw_label_streams):
        
        (prev_sublabels, next_sublabels) = (torch.LongTensor([0]),torch.LongTensor([0]))    #VARIABLE x2
        fw_streams = (fw_label_streams, prev_sublabels)
        bw_streams = (bw_label_streams, next_sublabels)
        
        #(sequence_length, batch_size, num_features) = input.size()
        #padder = torch.cuda.FloatTensor(1, batch_size, num_features)
        #for i in range( batch_size ):
        #    padder[:,i,:] = self.speaker_val[i]
        #padded_input = torch.cat( [padder, input, padder], 0 )

        self.slu_decoder.set_batch_size( batch_size )
        (fw_tk_scores, bw_tk_scores, tk_reps) = self.speech_decoder(input, bw_label_streams, fw_label_streams)
        self.slu_decoder.init_hidden()
        (sem_hidden_states, sem_scores) = self.slu_decoder([tk_reps], fw_streams)

        return (sem_scores, sem_scores, sem_hidden_states)

    def load_speech_encoder(self, speech_encoder):

        self.speech_decoder.load_state_dict( speech_encoder.state_dict() )
        if isinstance(speech_encoder, MLSpeechSeqEncoder):
            print(' -- SLUSimpleDecoder: freezing speech-encoder parameters...')
            sys.stdout.flush()
            for param in self.speech_decoder.char_encoder.parameters():
                param.requires_grad = False

    def set_speaker_val(self, val):
    
        self.speaker_val = val
    
    def pad_input(self, input, val):
    
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

    def set_test_mode(self, val):
        
        self.speech_decoder.set_test_mode( val )
        self.slu_decoder.set_test_mode( val )

    def set_teaching_signal_flag(self, val):
    
        self.speech_decoder.set_teaching_signal_flag( val )
        self.slu_decoder.set_teaching_signal_flag( val )

class SLUBiDecoder(nn.Module):
    
    def __init__(self, ch_params, tk_params, nn_params):
        
        super(SLUBiDecoder,self).__init__()
        
        self.speaker_val = [globals.user_speaker_val]
        
        tmp = nn_params.tag_vocab_size
        nn_params.tag_vocab_size = nn_params.sd_tag_vocab_size
        decoder_output_size = 0
        if nn_params.train_char_decoder or nn_params.load_char_decoder:
            print(' -- SLUBiDecoder: using character speech decoder')
            sys.stdout.flush()
            self.speech_decoder = BasicSpeechSeqEncoder(ch_params, nn_params)
            decoder_output_size = nn_params.hidden_dim
        elif nn_params.train_token_decoder or nn_params.load_token_decoder:
            print(' -- SLUBiDecoder: using token speech decoder')
            sys.stdout.flush()
            self.speech_decoder = BasicSpeechSeqEncoder(tk_params, nn_params)
            decoder_output_size = nn_params.hidden_dim
        elif nn_params.train_ml_decoder or nn_params.load_ml_decoder:
            print(' -- SLUBiDecoder: using 2-stage token speech decoder')
            sys.stdout.flush()
            self.speech_decoder = MLSpeechSeqEncoder(ch_params, tk_params, nn_params)
            decoder_output_size = nn_params.hidden_dim

        nn_params.tag_vocab_size = tmp
        nn_params.label_embed_dim = 2 * nn_params.label_embed_dim
        nn_params.hidden_dim = 2 * nn_params.hidden_dim
        self.bw_slu_decoder = SimpleDecoder(nn_params, decoder_output_size, 1)
        self.fw_slu_decoder = BidirectionalDecoder(nn_params, decoder_output_size, 0)

    def forward(self, input, bw_label_streams, fw_label_streams):
        
        (prev_sublabels, next_sublabels) = (torch.LongTensor([0]),torch.LongTensor([0]))    #VARIABLE x2
        fw_streams = (fw_label_streams, prev_sublabels)
        bw_streams = (bw_label_streams, next_sublabels) 
        
        self.bw_slu_decoder.set_batch_size( batch_size )
        self.fw_slu_decoder.set_batch_size( batch_size )
        (fw_tk_scores, bw_tk_scores, tk_reps) = self.speech_decoder(input, bw_label_streams, fw_label_streams)
        self.bw_slu_decoder.init_hidden()
        self.fw_slu_decoder.init_hidden()
        (sem_bw_hidden_states, sem_bw_scores) = self.bw_slu_decoder([tk_reps], bw_streams)
        (sem_fw_hidden_states, sem_fw_scores) = self.fw_slu_decoder([tk_reps], fw_streams, [sem_bw_hidden_states, sem_bw_scores])
        global_scores = 0.5 * (sem_fw_scores + sem_bw_scores)

        return (global_scores, sem_bw_scores, sem_hidden_states)

    def load_speech_encoder(self, speech_encoder):
        
        self.speech_decoder.load_state_dict( speech_encoder.state_dict() )
        if isinstance(speech_encoder, MLSpeechSeqEncoder):
            print(' -- SLUBiDecoder: freezing speech-encoder parameters...')
            sys.stdout.flush()
            for param in self.speech_decoder.char_encoder.parameters():
                param.requires_grad = False

    def set_speaker_val(self, val):
        
        self.speaker_val = val
    
    def pad_input(self, input, val):
        
        self.speaker_val = val
        (sequence_length, batch_size, num_features) = input.size()
        padder = torch.FloatTensor(1, batch_size, num_features).to(input.device)
        for i in range( batch_size ):
            padder[:,i,:] = self.speaker_val[i]
        return torch.cat( [padder, input, padder], 0 )

    def set_test_mode(self, val):
        
        self.speech_decoder.set_test_mode( val )
        self.slu_decoder.set_test_mode( val )

    def set_teaching_signal_flag(self, val):
        
        self.speech_decoder.set_teaching_signal_flag( val )
        self.slu_decoder.set_teaching_signal_flag( val )

