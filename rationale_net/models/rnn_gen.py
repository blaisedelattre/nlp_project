import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class RNN(nn.Module):
    def __init__(self, args, max_pool_over_time=False):
        super(RNN, self).__init__()
        self.args = args

        
        input_size = self.args.embedding_dim

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=args.hidden_dim,
                        batch_first=True, dropout=args.dropout, num_layers=1, # 1 layer seems the best ( better than 2)
                        bidirectional=True)

        # skip or not, 2 classes
        in_features = args.hidden_dim
        if args.gen_bidirectional:
            in_features *= 2
        self.hidden = nn.Linear(in_features=in_features, out_features=2, bias=True)
    
    def forward(self, x):
        # outputs: [batch_size, max_steps, hidden_dim]
        outputs, (h_n, c_n) = self.rnn(x)
        logits = self.hidden(outputs)
        return logits





