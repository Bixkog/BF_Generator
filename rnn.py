import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class BFgen(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=1, batch_size=1):
        super(BFgen, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.functional.log_softmax
        
    def forward(self, input_token):
        embeds = self.encoder(input_token)
        output, self.hidden = self.lstm(
            embeds.view(len(input_token), self.batch_size, -1), self.hidden)
        output = self.decoder(output.view(self.batch_size, -1))
        output = self.softmax(output)
        return output
    
    def init_hidden_zero(self):
        self.hidden = (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),)
    
    def init_hidden_normal(self):
        means = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        std = torch.Tensor([0.001]*self.hidden_size*self.n_layers*self.batch_size).unsqueeze(0)
        self.hidden = (Variable(torch.normal(means, std)), Variable(torch.normal(means, std)))


# embedding_size = 10
# hidden_size = 35
# output_size = 9
# n_layers = 2