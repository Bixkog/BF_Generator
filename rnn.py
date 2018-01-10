import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

token = {
    '.' : 0, 
    ',' : 1, 
    '[' : 2, 
    ']' : 3, 
    '<' : 4, 
    '>' : 5, 
    '+' : 6, 
    '-' : 7,
    "START" : 8
    }

char = {
    0 : '.',
    1 : ',', 
    2 : '[', 
    3 : ']',  
    4 : '<',  
    5 : '>', 
    6 : '+',  
    7 : '-'
    # no START on purpose
    }

embedding_size = 10
hidden_size = 35
output_size = len(char.keys())
n_layers = 2
batch_size = 16
token_num = len(token.keys())

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
        self.softmax = nn.functional.softmax
        

    """
    forward
    Takes input_token and hidden memory state <- input to recursive layer
    returns output token and changed hidden memory state.
    """
    def forward(self, input_token):
        embeds = self.encoder(input_token)
        output, self.hidden = self.lstm(embeds, self.hidden)
        decoded = self.decoder(output)
        probs = self.softmax(decoded[-1])
        return probs
    
    def init_hidden_zero(self):
        """
        Initializes hidden state and cell state for LSTM as zeros
        """
        self.hidden = (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),)
    
    def init_hidden_normal(self, variance=0.01):
        """
        Initialzes hidden state and cell state for LSTM with values from Normal Distribution ~ N(0, variance)
        """
        means = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        std = torch.Tensor([variance]*self.hidden_size*self.n_layers*self.batch_size).unsqueeze(0)
        self.hidden = (Variable(torch.normal(means, std)), Variable(torch.normal(means, std)))

def token_to_tensor(input_token):
    """
    Takes one of BF language tokens, returns its one-hot vector as torch.LongTensor
    Args:
        input_token(string): BF language token
    Returns:
        tensor(torch.LongTensor): token's one-hot vector 
    """
    tensor = torch.zeros(1, token_num).long()
    tensor[0][token[input_token]] = 1
    return tensor

def pred(sample):
    """
    Takes np.array with probabilites, returns the highest probability
    """
    return np.argsort(sample)[-1]


def evaluate(model, predict_len=100, variance=0.01):
"""
    Evaluate given model
    Args:
        model(nn.Module): given neural net model
        predict_len(int): length of programs to generate
        variance(float): a variance for initialization of LSTM's hidden state,
            which values are taken from N(0, variance)
"""
    input_token = "START"
    input_token = token_to_tensor(input_token)
    model.init_hidden_normal(variance=0.5)
    prediction = [""] * batch_size
    program_probs = np.ones((1, batch_size))
    
    batched_input = torch.zeros((batch_size, token_num)).long()
    batched_input = batched_input + input_token
    batched_input = Variable(batched_input.view(token_num, batch_size))
    
    for i in range(predict_len):
        output_probs = model.forward(batched_input)
        next_tokens = np.apply_along_axis(pred, axis = 1, arr=output_probs.data.numpy())
        top_probs = output_probs[np.arange(batch_size), next_tokens]
        
        batched_input = torch.zeros((batch_size, token_num)).long()
        batched_input[np.arange(batch_size), next_tokens] = 1
        batched_input = Variable(batched_input.view(token_num, batch_size))
        
        program_probs *= top_probs.data.numpy()
        
        for i in xrange(batch_size):
            prediction[i] += char[next_tokens[i]]
        
    return prediction, program_probs
