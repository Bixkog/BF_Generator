import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

token = 
    {
    '\0' : 0,
    '.' : 1, 
    ',' : 2, 
    '[' : 3, 
    ']' : 4, 
    '<' : 5, 
    '>' : 6, 
    '+' : 7, 
    '-' : 8,
    "START" : 9
    }

char = 
    {
    0 : '\0',
    1 : '.',
    2 : ',', 
    3 : '[', 
    4 : ']',  
    5 : '<',  
    6 : '>', 
    7 : '+',  
    8 : '-'
    # no START on purpose
    }

class BFgen(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=2, batch_size=1):
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
        

    """
    forward
    Takes input_token and hidden memory state <- input to recursive layer
    returns output token and changed hidden memory state.
    """
    def forward(self, input_token, hidden):
        embeds = self.encoder(input_token)
        output, hidden = self.lstm(
            embeds.view(len(input_token), self.batch_size, -1), hidden)
        output = self.decoder(output.view(self.batch_size, -1))
        output = self.softmax(output) # in paper its multinomial distribution
        return output, hidden
    
    def init_hidden_zero(self):
        self.hidden = (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),)
    
    def init_hidden_normal(self):
        means = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        std = torch.Tensor([0.001]*self.hidden_size*self.n_layers*self.batch_size).unsqueeze(0)
        self.hidden = (Variable(torch.normal(means, std)), Variable(torch.normal(means, std)))

    def evaluate(self, predict_len=100):
        input_token = token["START"]
        hidden = self.init_hidden_zero
        prediction = ""

        for i in range(predict_len):
            output_token, hidden = self.forward(input_token, hidden)
            input_token = output_token

            prediction += char[output_token]
            if output_token == '\0':
                break

        return prediction





# embedding_size = 10
# hidden_size = 35
# output_size = 9
# n_layers = 2