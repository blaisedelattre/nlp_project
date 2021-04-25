import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        input_size = args.embedding_dim
        self.lstm = nn.LSTM(input_size, args.hidden_dim, args.num_layers)
        self.linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout_final = nn.Dropout(args.dropout)

    def forward(self, x, mask=None, length=None):
        if not mask is None:
            x = x * mask.unsqueeze(-1)
        # initial state always zero
        output, (_, _) = self.lstm(x)
        # take the final hidden state and use it for prediction
        last_output = output[:,-1,:]
        last_output = self.dropout_final(last_output)
        # sert surement pas à grand chose
        last_output = self.linear(last_output)
        return last_output.squeeze(1)