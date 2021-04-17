# Code for adding the End2End SLU task into fairseq

import os
import re
import sys
import torch
import math

from fairseq.globals import *
from fairseq.data import Dictionary, End2EndSLUDataset
from fairseq.tasks import FairseqTask, register_task

import examples.speech_recognition.criterions.SLU_CTC_loss as slu_ctc

class SLUDictionary(Dictionary):
    """Dictionary wrapping to initialize a dictionary from a raw python dictionary"""

    def __init__(
        self,
        pad=pad_token,
        eos=EOS_tag,
        unk=unk_token,
        bos=SOS_tag,
        extra_special_symbols=None,
    ):
        super(SLUDictionary, self).__init__(pad, eos, unk, bos, extra_special_symbols)
        # The above statement does:
        #   1. self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        #   2.  self.bos_index = self.add_symbol(bos)
        #       self.pad_index = self.add_symbol(pad)
        #       self.eos_index = self.add_symbol(eos)
        #       self.unk_index = self.add_symbol(unk)
        #   3. Add the other special-symbols

        self.bos_word = bos

    def init_from_python_dict(self, dict):

        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = 0

        for v, k in enumerate(dict):
            self.add_symbol(k)

        self.bos_index = self.add_symbol(self.bos_word)
        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)

    def extend_from_python_dict(self, dict):
        for v, k in enumerate(dict):
            self.add_symbol(k)

        # just in case...
        self.bos_index = self.add_symbol(self.bos_word)
        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)

    def reset(self):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = 0
        self.bos_word = self.pad_word = self.eos_word = self.unk_word = ''
        self.bos_index = self.pad_index = self.eos_index = self.unk_index = -1

    def set_nspecial_(self):
        self.nspecial = len(self.symbols)

    def set_bos_(self, bos=SOS_tag):
        self.bos_word = bos
        self.bos_index = self.add_symbol(self.bos_word)
        return self.bos_index

    def set_pad_(self,pad=pad_token):
        self.pad_word = pad
        self.pad_index = self.add_symbol(self.pad_word)
        return self.pad_index

    def set_eos_(self, eos=EOS_tag):
        self.eos_word = eos
        self.eos_index = self.add_symbol(self.eos_word)
        return self.eos_index

    def set_unk_(self, unk=unk_token):
        self.unk_word = unk
        self.unk_index = self.add_symbol(self.unk_word)
        return self.unk_index

    def set_blank(self, blank=blank_token):
        self.blank_word = blank
        return self.add_symbol(blank)

def create_batch_(data, curr_batch_ids):
    
    dialog_len = len( data[curr_batch_ids[0]] ) # all dialogs in this batch have this same length
    dialog_batch = []
    for i in range( dialog_len ):
        for d_idx in range(len(curr_batch_ids)):

            dialog_batch.append( (curr_batch_ids[d_idx], i) )

    return dialog_batch

def create_dialog_batches_(data, batch_size):
    
    dialog_lengths = {}
    for dialog_id in data.keys():
        curr_len = len(data[dialog_id])
        if not curr_len in dialog_lengths:
            dialog_lengths[curr_len] = []
        dialog_lengths[curr_len].append( dialog_id )

    dialog_batches = []
    for dlen in dialog_lengths.keys():
        
        if len(dialog_lengths[dlen]) > 1:
            
            num_batches = int(len(dialog_lengths[dlen]) / batch_size)
            remainder = len(dialog_lengths[dlen]) % batch_size
            
            b_idx = 0
            while b_idx < num_batches:
                curr_batch_ids = dialog_lengths[dlen][b_idx*batch_size:(b_idx+1)*batch_size]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
                b_idx += 1
            
            if remainder > 0:
                curr_batch_ids = dialog_lengths[dlen][num_batches*batch_size:]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
        else:
            curr_batch_ids = dialog_lengths[dlen]
            dialog_batches.append( create_batch_(data, curr_batch_ids) )

    return dialog_batches

def read_txt(txtpath):
 
    f = open(txtpath, 'rb')
    lines = [l.decode('utf-8','ignore') for l in f.readlines()]
    for line in lines:
        dialog_transcript = line
    f.close()
    return dialog_transcript.strip()

def parse_media_semantic_annotation(filename):
    
    # Example:
    # 364_16 @null{euh} @+rang-temps{la troisième} @+temps-unite{semaine} @+temps-mois{d' août}
    
    clean_re = re.compile('^\s+|\s+$')
    p = re.compile('^(\+|\-)?(\S+)\{([^\}]+)\}')
    
    Dialogs = {}
    f = open(filename)
    for line in f:
        line.rstrip("\r\n")
        tokens = line.split()
        ID = tokens[0]
        #(dialog_ID,turn_ID) = ID.split("_")
        # TODO: check that the following lines of code, up to concepts_str..., are generic with any corpus ID!
        ID_tokens = ID.split("_")
        dialog_ID = "_".join(ID_tokens[:-1])
        turn_ID = ID_tokens[-1]
        concepts_str = clean_re.sub('', ' '.join(tokens[1:]))

        print(' - parse_media_semantic_annotation, dialog_ID and turn_ID: {}, {}'.format(dialog_ID, turn_ID))
        sys.stdout.flush()

        TurnStruct = {}
        TurnStruct['ID'] = turn_ID
        if not dialog_ID in Dialogs:
            Dialogs[dialog_ID] = []

        concept_list = []
        concepts = concepts_str.split('@')
        for c in concepts:
            if len(c) > 0:
                m = p.match(c)
                if m == None:
                    sys.stderr.write(' - ERROR: parse_media_semantic_annotation parsing error at {}\n'.format(c))
                    sys.exit(1)
                else:
                    (mode,concept,surface) = (m.group(1),m.group(2),clean_re.sub('',m.group(3)))
                    concept_list.append( (mode,concept,surface) )
        TurnStruct['Concepts'] = concept_list
        Dialogs[dialog_ID].append( TurnStruct )

    f.close() 
    return Dialogs

def parse_media_filename( turn ):

    file_prefix = turn.split('/')[-1]   # e.g. dialog-611.turn-26-Machine-overlap_none.channel-1
    dialog_id = file_prefix.split('.')[0]
    turn_id = file_prefix.split('.')[1]
    turn_id_comps = turn_id.split('-')[0:3]
    turn_id = '-'.join(turn_id_comps)

    return dialog_id, turn_id

def parse_fsc_filename( turn ):

    components = turn.split('/')    # e.g. .../2BqVo8kVB2Skwgyb/029f6450-447a-11e9-a9a5-5dbec3b8816a
    dialog_id = components[-2]
    turn_id = components[-1]

    return dialog_id, turn_id

def parse_filename( args, turn ):

    if args.corpus_name in ['media', 'etape']:
        return parse_media_filename(turn)
    elif args.corpus_name == 'fsc':
        return parse_fsc_filename(turn)
    else:
        raise NotImplementedError

def read_dialog_data(TurnList, args):
    
    windowtime = args.window_time 
    reduce_flag = args.reduce_signal
    lan_flag = args.w2v_language

    if windowtime < 0:
        srate_str = '16kHz'
        if windowtime == -2:
            srate_str = '8kHz'
        if windowtime > -3:
            print(' * read_dialog_data: reading {} wave2vec features...'.format(srate_str))
        elif windowtime == -3:
            print(' * read_dialog_data: reading FlowBERT features (.fbert files)...')
        elif windowtime == -4:
            print(' * read_dialog_data: reading W2V2 base features (.bebert files)...')
        elif windowtime == -5:
            print(' * read_dialog_data: reading FlowBERT base features (.bbert files)...')
        elif windowtime == -6:
            print(' * read_dialog_data: reading W2V2 large features (.ebert files)...')
        elif windowtime == -7:
            print(' * read_dialog_data: reading XLSR53 large features (.xbert files)...')
        else:
            raise NotImplementedError()
        sys.stdout.flush()

    dialog_data = {}
    with open(TurnList) as f:
        turns = f.readlines()
        f.close()

    for turn in turns:
        dialog_id, turn_id = parse_filename( args, turn )

        if dialog_id not in dialog_data:
            dialog_data[dialog_id] = [] 

        # 1. Turn ID for sorting turns of the same dialog
        turn_data = []
        turn_data.append( turn_id )

        # 2. Spectrogram of the turn audio signal (the input to the network)
        if windowtime == 20:
            file_ext = '.8kHz.20.0ms-spg'
            spg_tsr = torch.load(turn.strip() + file_ext)
        elif windowtime == 10:
            raise NotImplementedError
        elif windowtime < 0:
            srate_str = '16kHz'
            if windowtime == -2:
                srate_str = '8kHz'
            lan=''
            if lan_flag == 'Fr':
                lan='-Fr'
            if windowtime > -3:
                spg_tsr = torch.load(turn.strip() + '.' + srate_str + lan + '.w2v')
            elif windowtime == -3:
                spg_tsr = torch.load(turn.strip() + '.fbert')
            elif windowtime == -4: 
                spg_tsr = torch.load(turn.strip() + '.ebert')
            elif windowtime == -5:
                spg_tsr = torch.load(turn.strip() + '.bbert')
            elif windowtime == -6:
                spg_tsr = torch.load(turn.strip() + '.bebert')
            elif windowtime == -7:
                spg_tsr = torch.load(turn.strip() + '.xbert')
            if reduce_flag > 1:
                raise NotImplementedError

        if windowtime > -3:
            spg_tsr = spg_tsr.squeeze().permute(1,0)
        else:
            spg_tsr = spg_tsr.squeeze() 
        if len(spg_tsr.size()) != 2:
            spg_tsr = spg_tsr.unsqueeze(0)

        if spg_tsr.size(0) < 3:
            print(' *** read_dialog_data: got strangely short signal {}...'.format(spg_tsr.size()))
            sys.stdout.flush()

        if spg_tsr.size(0) > 3 and spg_tsr.size(0) < args.max_source_positions:
            turn_data.append( spg_tsr )  # Spectrogram's shape is (1 x num_features x sequence_length), we make it (sequence_length x num_features) 

            # 3. The reference transcription 
            turn_txt = read_txt(turn.strip() + '.txt')
            turn_data.append( turn_txt ) 

            # 4. The transcription "enriched" with semantic annotation
            basename = turn.split('/')[-1]
            if args.corpus_name in ['media', 'etape']:
                if user_ID in basename:
                    dialog_struct = parse_media_semantic_annotation( turn.strip() + '.sem' )
                    slu_turn = ''
                    for vals in dialog_struct.values():
                        for turn_struct in vals: 
                            for c in turn_struct['Concepts']:
                                if len(slu_turn) > 0:
                                    slu_turn = slu_turn + ' '
                                slu_turn = slu_turn + slu_start_concept_mark + ' ' + c[2] + ' ' + c[1] + ' ' + slu_end_concept_mark
                    turn_data.append( slu_turn )
                    turn_data.append( user_ID )
                else:
                    slu_turn = slu_start_concept_mark + ' ' + turn_txt + ' ' + machine_semantic + ' ' + slu_end_concept_mark
                    turn_data.append( slu_turn )
                    turn_data.append( machine_ID )

                if turn_id in dialog_data[dialog_id]:
                    sys.stderr.write(' *** read_dialog_data WARNING: turn_id {} is not unic\n'.format(turn_id))
            elif args.corpus_name == 'fsc':
                turn_sem = read_txt(turn.strip() + '.sem')
                turn_data.append( turn_sem )
            else:
                raise NotImplementedError

            dialog_data[dialog_id].append( turn_data )

    return dialog_data

def references2indexes(args, data, vocab):

    for did in data.keys():
        for idx in range(len(data[did])):
            (turn_id, spg, txt_turn, slu_turn, spk_ID) = data[did][idx]
            char_list = [vocab.add_symbol(c) for c in txt_turn]
            token_list = [vocab.add_symbol(t) for t in txt_turn.split()]
            slu_list = [vocab.add_symbol(s) for s in slu_turn.split()]

            if args.padded_reference:
                char_list = [vocab.bos()] + char_list + [vocab.eos()]
                token_list = [vocab.bos()] + token_list + [vocab.eos()]
                slu_list = [vocab.bos()] + slu_list + [vocab.eos()]

            char_tsr = torch.LongTensor( char_list )
            token_tsr = torch.LongTensor( token_list )
            slu_tsr = torch.LongTensor( slu_list )

            data[did][idx] = (turn_id, spg, char_tsr, token_tsr, slu_tsr, spk_ID)


def feature_wise_mean_std(data):
    (sequence_length, num_features) = list(data.values())[0][0][1].size()
    data_mu = torch.zeros( num_features )
    data_sigma = torch.zeros( num_features )

    with torch.no_grad():
        total_features = 0
        for dialog_id in data.keys():
            for t in data[dialog_id]:
                data_mu = data_mu + torch.sum( t[1], 0 )
                total_features += t[1].size(0)
        data_mu = data_mu / float( total_features )

        for dialog_id in data.keys():
            for t in data[dialog_id]:
                data_sigma = data_sigma + torch.sum( (t[1] - data_mu)**2, 0 )
        data_sigma = torch.sqrt( data_sigma / float(total_features-1) )

    return (data_mu, data_sigma)

def normalize_data_only(data, mu, sigma):
    with torch.no_grad():
        for dialog_id in data.keys():
            for t in data[dialog_id]:
                t[1] = (t[1] - mu) / sigma


@register_task('end2end_slu')
class End2EndSLU(FairseqTask):
    
    @staticmethod
    def add_args(parser):
        
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--serialized-data', type=str,
                            help='file containing the serialized corpus (with a previous torch.save)')
        parser.add_argument('--max-source-positions', default=10000, type=int,
                            help='max input length')
        parser.add_argument('--max-target-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--slu-subtask', default='char', type=str,
                            help='which subtask has to be modeled (e.g. char, token, concept)')
        parser.add_argument('--user-only', action='store_true',
                            help='use only user turns (specific to SLU for dialog systems, and to the corpus MEDIA in particular)')
        parser.add_argument("--padded-reference", action='store_true', default=False,
                            help="Specify if the gold reference is padded")
        parser.add_argument('--test-mode', action='store_true', default=False,
                            help='Specify that model must be run in test mode')
        parser.add_argument('--reduce-signal', type=int, default=0,
                            help='Reduce the input signal by the given factor (only for wav2vec features)')
        parser.add_argument('--window-time', type=int, default=20,
                            help='Specify the window time, in milliseconds, used for extracting spectrograms. If negative, wav2vec features are used.')
        parser.add_argument('--w2v-language', type=str, default='En',
                            help='Specify which languege wav2vec features are from')
        parser.add_argument('--mode', type=str, default='user+machine',
                            help='Leave this unchanged!')
        parser.add_argument('--io-size-ratio', type=int, default=5,
                            help='The estimated ratio between input and output sequence lenghts')
        parser.add_argument('--bidirectional-encoder', action='store_true', default=True,
                            help='Use a bidirectional LSTM in the encoder')
        parser.add_argument('--encoder-transformer-layers', action='store_true', default=False,
                            help='Use transformer encoder layers on top of the convolution+recurrent encoder')
        parser.add_argument('--corpus-name', type=str, default='media',
                            help='Specify the corpus name to customize the data reading. 1) media (default), 2) fsc (fluent speech commands)')
        parser.add_argument( '--load-encoder', type=str, help='Load a pre-trained (basic) encoder' )
        parser.add_argument( '--load-fairseq-encoder', type=str, help='Load the encoder from a fairseq checkpoint' )
        parser.add_argument( '--slu-end2end', action='store_true', default=False, help='Add an auxiliary loss for an additional output expected in the model output structure' )
        parser.add_argument('--scheduled-sampling', action='store_true', default=False, help='Use scheduled sampling during training')

    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just initialize the label dictionary
        
        label_vocab = SLUDictionary(
            pad=pad_token,
            eos=EOS_tag,
            unk=unk_token,
            bos=SOS_tag,
            extra_special_symbols=[blank_token, machine_semantic, slu_start_concept_mark, slu_end_concept_mark]
        )
        # NOTE: we are ogliged to initialize the Dictionary as it is requested in Fairseq,
        #       but then we reset it and re-add symbols so that to have index 0 for the blank token, as needed by the CTC loss.
        label_vocab.reset() 
        label_vocab.set_blank(blank_token)
        label_vocab.set_pad_(pad_token)
        label_vocab.set_eos_(EOS_tag)
        label_vocab.set_unk_(unk_token)
        label_vocab.set_bos_(SOS_tag)
 
        label_vocab.add_symbol(machine_semantic)
        label_vocab.add_symbol(slu_start_concept_mark)
        label_vocab.add_symbol(slu_end_concept_mark)
        label_vocab.set_nspecial_()  # We set the tokens added so far as being special tokens.

        print('| [label] dictionary: {} types'.format(len(label_vocab)))
        
        return End2EndSLU(args, label_vocab)

    def __init__(self, args, label_vocab):
        
        super().__init__(args)
        self.label_vocab = label_vocab
        self.num_features = 81

        self.blank_idx = label_vocab.index( blank_token )
        self.sos_tag_idx = label_vocab.index( SOS_tag )
        self.eos_tag_idx = label_vocab.index( EOS_tag )
        self.slu_start_concept_idx = label_vocab.index( slu_start_concept_mark )
        self.slu_end_concept_idx = label_vocab.index( slu_end_concept_mark )

        # Scheduled sampling state and needed values
        self.criterion = args.criterion
        self.scheduled_sampling = args.scheduled_sampling
        if self.scheduled_sampling:
            print(' - End2EndSLU task: using scheduled sampling')
            sys.stdout.flush()
        self.nsteps = 0
        self.best_train_loss = LOSS_INIT_VALUE
        self.curr_train_loss = LOSS_INIT_VALUE
        self.best_dev_loss = LOSS_INIT_VALUE
        self.curr_dev_loss = LOSS_INIT_VALUE
        self.dev_errors = 0
        self.dev_tokens = 0
        self.best_dev_er = ER_INIT_VALUE
        self.curr_dev_er = ER_INIT_VALUE
        self.tf_tries = 0
        self.ss_tries = 0
        self.tf_freedom_tries = 0
        self.ss_freedom_tries = 0
        self.switch_to_come = False
        self.init_ss_threshold = 0.99

        self.curr_epoch = 0
        self.max_tf_epochs = 5          # Max # of epochs with teacher forcing
        self.max_ss_epochs = 5          # Max # of epochs with scheduled sampling
        self.tf_patience = 3            # # of epochs of tf training without improvement
        self.ss_patience = 3            # Same for scheduled sampling
        self.forced_switch_epoch = 2    # Force switching between tf and ss training after this number of epochs at the beginning of training

    def set_scheduled_sampling(self, ss_val=False):
        self.scheduled_sampling = ss_val

    def switch_scheduled_sampling(self):
        self.scheduled_sampling = not self.scheduled_sampling

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
            
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                    gradient
                - logging outputs to display while training
        """
        
        loss, sample_size, logging_output = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        self.curr_train_loss += loss.item()
        self.nsteps += 1

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        self.curr_dev_loss += loss.item()
        if isinstance(criterion, slu_ctc.SLUCTCCriterion):
            self.dev_errors += logging_output['errors']
            self.dev_tokens += logging_output['total']

        return loss, sample, logging_output

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""

        n_ss_up = 0
        self.curr_epoch += 1
        if self.curr_epoch <= 3:
            self.datasets['train'].curriculum(value=True)
        if self.curr_epoch > 3:
            self.datasets['train'].curriculum(value=False)

        if epoch > 1:
            self.init_ss_threshold *= 0.98
            if self.scheduled_sampling:
                n_ss_up = model.get_scheduled_sampling_updates()
                model.set_scheduled_sampling(ss_val=True, epoch=epoch)

            if self.nsteps > 0: # if we reload a checkpoint, epoch can be > 1, but nsteps is 0
                self.curr_train_loss /= self.nsteps
                self.curr_dev_loss /= self.nsteps
            if self.dev_tokens > 0:
                self.curr_dev_er = float(self.dev_errors) / float(self.dev_tokens)

            if self.criterion == 'slu_ctc_loss':
                print(' ----------')
                if self.scheduled_sampling:
                    print(' Dev error rate summary after epoch {} (scheduled sampling updates: {})'.format(epoch-1, n_ss_up))
                else:
                    print(' Dev error rate summary after epoch {}'.format(epoch-1))
                print(' * Current dev error rate: {:.4f} (best {:.4f})'.format(self.curr_dev_er, self.best_dev_er))
                print(' ----------')
                sys.stdout.flush()

            if self.best_train_loss > self.curr_train_loss:
                self.best_train_loss = self.curr_train_loss
            if self.best_dev_loss > self.curr_dev_loss:
                self.best_dev_loss = self.curr_dev_loss
            if self.best_dev_er > self.curr_dev_er:
                self.best_dev_er = self.curr_dev_er

        self.curr_train_loss = 0.0
        self.curr_dev_loss = 0.0
        self.curr_dev_er = 0.0
        self.dev_errors = 0
        self.dev_tokens = 0
        self.nsteps = 0

    def add_speaker_marker(self, feats, spk):
        """
            Add leading and trailing vectors as speaker marker for distinguishing input features from different speakers.
            When using the MEDIA corpus it is important to distinguish between the Machine, which is much easier to recognize, and a User.
        """

        spk_abs_val = 5.0
        (src_len, dim) = feats.size()
        spk_val = 0.0
        if spk == user_ID:
            spk_val = +spk_abs_val
        elif spk == machine_ID:
            spk_val = -spk_abs_val
        else:
            raise NotImplementedError

        padder = torch.zeros(3, dim).fill_(spk_val).to(feats)
        return torch.cat( [padder, feats, padder], 0 )

    def load_dataset(self, split, **kwargs):
    
        """Load a given dataset split (e.g., train, valid, test)."""

        corpus = {}
        my_split = split
        if my_split == 'valid':
            my_split = 'dev'    # For compatibility with the old End2End SLU system (ICASSP 2020) 

        if os.path.exists(self.args.serialized_data + '.' + my_split) and os.path.isfile(self.args.serialized_data + '.' + my_split):        
            print(' - Loading serialized {} split...'.format(my_split))
            sys.stdout.flush()

            corpus[my_split] = torch.load(self.args.serialized_data + '.' + my_split)
        else:
            print(' - Reading dialog data')
            print('   - reading train data...')
            sys.stdout.flush()
            train_data = read_dialog_data( self.args.data + '/train_prefixes.lst', self.args )

            print('   - reading dev data...')
            sys.stdout.flush()
            dev_data = read_dialog_data( self.args.data + '/dev_prefixes.lst', self.args )

            print('   - reading test data...')
            sys.stdout.flush()
            test_data = read_dialog_data( self.args.data + '/test_prefixes.lst', self.args )

            print( ' - Normalizing input data...' )
            sys.stdout.flush()
            (train_mu, train_sigma) = feature_wise_mean_std( train_data )
            normalize_data_only(train_data, train_mu, train_sigma)
            normalize_data_only(dev_data, train_mu, train_sigma)
            normalize_data_only(test_data, train_mu, train_sigma)

            print(' - Mapping tokens to indexes...')
            sys.stdout.flush()
            references2indexes(self.args, train_data, self.label_vocab)
            references2indexes(self.args, dev_data, self.label_vocab)
            references2indexes(self.args, test_data, self.label_vocab)
            print('   - Dictionary size: {}'.format(len(self.label_vocab)))
            sys.stdout.flush()

            corpus = {}
            corpus['train'] = train_data
            corpus['dev'] = dev_data
            corpus['test'] = test_data
            corpus['dictionary'] = self.label_vocab

            print(' - Saving serialized corpus...')
            sys.stdout.flush()
            torch.save(train_data, self.args.serialized_data + '.train' )
            torch.save(dev_data, self.args.serialized_data + '.dev' )
            torch.save(test_data, self.args.serialized_data + '.test' )
            torch.save(self.label_vocab, self.args.serialized_data + '.dict' )


        if not 'dictionary' in corpus:
            corpus['dictionary'] = torch.load(self.args.serialized_data + '.dict')
            self.label_vocab = corpus['dictionary'] 
            print(' - Loaded dictionary size: {}'.format(len(self.label_vocab)))
            sys.stdout.flush()

        if my_split == 'train' and 'test' in corpus:
            corpus['test'] = None
        self.blank_idx = self.label_vocab.index( blank_token )
        self.sos_tag_idx = self.label_vocab.index( SOS_tag )
        self.eos_tag_idx = self.label_vocab.index( EOS_tag )
        self.slu_start_concept_idx = self.label_vocab.index( slu_start_concept_mark )
        self.slu_end_concept_idx = self.label_vocab.index( slu_end_concept_mark ) 

        # data for each turn are: turn-id, signal-tensor, char-sequence (int) tensor, token-sequence (int) tensor, concept-sequence (int) tensor, speaker-id (machine or user)
        # data are organized as a dict of lists: the dict is indexed with dialog-id, the value is the list of turns for that dialog, each turn structure contains data described above.

        print(' - Reorganizing {} dialog data...'.format(my_split))
        if self.args.user_only:
            print('   - Loading user turns only.') 

        dialogs = corpus[my_split]
        bsz = 1
        if my_split == 'train':
            while( bsz < self.args.max_sentences ):
                bsz = bsz * 2
            if bsz > self.args.max_sentences:
                bsz = int(bsz / 2)
            if self.args.max_sentences == 5:
                bsz = 5
        batch_info = create_dialog_batches_(dialogs, bsz)
        idx_batches = []

        ratios = []
        sources = []
        src_lengths = []
        targets = []
        tgt_lengths = []
        global_turn_idx = 0
        for batch in batch_info:
            batched_turns_idx = []
            for (did, idx) in batch:
                turn = dialogs[did][idx]
                if (my_split == 'train') or (turn[-1] == user_ID):
                    feats = turn[1]
                    feats = self.add_speaker_marker(feats, turn[-1])
                    sources.append( feats )
                    src_lengths.append( feats.size(0) )

                    batched_turns_idx.append( (global_turn_idx, turn[-1]) )
                    global_turn_idx += 1

                    if self.args.slu_subtask == 'char':
                        targets.append( turn[2] )
                        tgt_lengths.append( turn[2].size(0) )

                        ratios.append( turn[2].size(0) / feats.size(0) )
                        
                    elif self.args.slu_subtask == 'token':
                        targets.append( turn[3] )
                        tgt_lengths.append( turn[3].size(0) )

                        ratios.append( turn[3].size(0) / feats.size(0) )
                        
                    elif self.args.slu_subtask == 'concept':
                        targets.append( turn[4] )
                        tgt_lengths.append( turn[4].size(0) )

                        ratios.append( turn[4].size(0) / feats.size(0) )

                    else:
                        raise NotImplementedError
            
            idx_batches.append( batched_turns_idx )

        print(' - Reorganized {} turns for split {}'.format(global_turn_idx, my_split))
        sys.stdout.flush()
        src_lengths_tsr = torch.FloatTensor( src_lengths )
        mean = torch.mean(src_lengths_tsr)
        median = torch.median(src_lengths_tsr)
        std = torch.std( src_lengths_tsr )
        max_len = torch.max( src_lengths_tsr )
        print(' - {} data statistics:'.format(my_split))
        print('   * Total # of turns: {}'.format(len(sources)))
        print('   * Mean source length: {:.2f}'.format(mean.item()))
        print('     - Std of source length: {:.2f}'.format(std.item()))
        print('     - Median source length: {}'.format(median.item()))
        print('     - Max. source length: {}'.format(max_len.item()))
        rmean = sum(ratios)/len(ratios)
        rstd = math.sqrt( sum([(el - rmean)**2 for el in ratios]) / (len(ratios)-1) )
        print('   * Mean target/source length ratio: {:.4f} (+/- {:.4f})'.format(rmean, rstd))
        print('   * Max. target/source length ratio: {:.4f}'.format(max(ratios)))
        print('   * Min. target/source length ratio: {:.4f}'.format(min(ratios)))
        sys.stdout.flush()
    
        # Reorganize turns based on increasing source length for curriculum learning
        src_info = [(i, sources[i].size(0)) for i in range(len(sources))]
        sorted_structure = sorted(src_info, key=lambda tuple: tuple[1])
        sources = [sources[t[0]] for t in sorted_structure]
        src_lengths = [src_lengths[t[0]] for t in sorted_structure]
        targets = [targets[t[0]] for t in sorted_structure]
        tgt_lengths = [tgt_lengths[t[0]] for t in sorted_structure]
        self.num_features = sources[0].size(-1)

        input_feed = True 
        self.datasets[split] = End2EndSLUDataset.End2EndSLUDataset(
            src=sources,
            src_sizes=src_lengths,
            tgt=targets,
            tgt_sizes=tgt_lengths,
            tgt_dict=self.label_vocab,
            idx_structure=idx_batches,
            left_pad_target=False,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions, 
            input_feeding=input_feed,
            shuffle=True,
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab


























