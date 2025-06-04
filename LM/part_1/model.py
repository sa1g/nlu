import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch  

class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()
        
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output
    
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output 