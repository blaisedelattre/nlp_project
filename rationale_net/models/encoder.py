import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import rationale_net.models.rnn_encoder as rnn
import rationale_net.models.rcnn as rcnn
import pdb

class Encoder(nn.Module):

    def __init__(self, embeddings, args):
        super(Encoder, self).__init__()
        ### Encoder
        self.args = args
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_dim = hidden_dim
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = True
        self.embedding_fc = nn.Linear( hidden_dim, hidden_dim )

        if args.model_form == 'cnn':
            # works for encoder
            self.cnn = cnn.CNN(args, max_pool_over_time=(not args.use_as_tagger))
            self.fc = nn.Linear( len(args.filters)*args.filter_num,  args.hidden_dim)
        elif args.model_form == 'rnn':
            # works for encoder
            self.rnn = rnn.RNN(args)
        elif args.model_form == 'rcnn':
            # works for encoder
            self.rcnn = rcnn.RCNN(args, embedding_dim=hidden_dim, hidden_size=hidden_dim, max_pool_over_time=True)
            self.fc = nn.Linear( hidden_dim,  args.hidden_dim)
        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, x_indx, mask=None, length=None):
        '''
            x_indx:  batch of word indices
            mask: Mask to apply over embeddings for tao ratioanles
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
                x = x.cuda()
        if self.args.model_form == 'cnn':
            if not mask is None:
                x = x * mask.unsqueeze(-1)

            x = F.relu( self.embedding_fc(x))

            x = self.dropout(x)
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = F.relu( self.fc(hidden) )

        elif self.args.model_form == 'rnn':
            # [batch_size, max_steps, embedding_size]
            hidden = self.rnn(x, length=length, mask=mask) # X in (Batch, Embed)
        elif self.args.model_form == 'rcnn':
            if not mask is None:
                x = x * mask.unsqueeze(-1)
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.rcnn(x) # X in (Batch, Embed)
            hidden = F.relu( self.fc(hidden) )
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(args.model_form))

        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return logit, hidden
