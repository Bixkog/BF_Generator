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

token_num = len(token.keys())

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

def token_to_tensor(input_token):
    tensor = torch.zeros(1, token_num).long()
    tensor[0][token[input_token]] = 1
    return tensor

def pred(sample):
    return np.argsort(sample)[-1]


class BFgen(nn.Module):
    def __init__(self, input_size, 
                       embedding_dim, 
                       hidden_size,
                       output_size, 
                       n_layers=2, 
                       batch_size=1,
                       GAMMA=0.99,
                       PG_learning_rate=1e-5,       # .\(ER) from paper
                       PQT_loss_multipiler=1.0,     # .\(TOPK) from paper
                       entropy_regularizer=0.01     # .\(ENT) from paper
                       ):
        super(BFgen, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.PG_learning_rate = PG_learning_rate
        self.PQT_loss_multipiler = PQT_loss_multipiler
        self.entropy_regularizer = entropy_regularizer
        
        self.encoder = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.functional.softmax
        
        self.baseline = 0.
        self.entropy = 0.
        
        self.pqt_programs = np.array([])
        self.pqt_rewards = np.array([])

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
        self.hidden = (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),)
    
    def init_hidden_normal(self, variance=0.01):
        means = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        std = torch.Tensor([variance]*self.hidden_size*self.n_layers*self.batch_size).unsqueeze(0)
        self.hidden = (Variable(torch.normal(means, std)), Variable(torch.normal(means, std)))

    def save_best(self, rewards, programs):
        self.pqt_programs = np.append(self.pqt_programs, programs)
        self.pqt_rewards = np.append(self.pqt_rewards, rewards)
        args = np.argsort(self.pqt_rewards)
        self.pqt_programs = self.pqt_programs[args][-10:]
        self.pqt_rewards = self.pqt_rewards[args][-10:] # K


def evaluate(model, predict_len=100, variance=0.01):
    input_token = "START"
    input_token = token_to_tensor(input_token)
    model.init_hidden_normal(variance=0.5)
    prediction = [""] * model.batch_size
    program_probs = np.ones((1, model.batch_size))
    
    batched_input = torch.zeros((model.batch_size, token_num)).long()
    batched_input = batched_input + input_token
    batched_input = Variable(batched_input.view(token_num, model.batch_size))
    
    for i in range(predict_len):
        output_probs = model.forward(batched_input)
        self.entropy = -(torch.sum(output_probs * torch.log(output_probs), 0))
        top_probs, next_tokens = torch.max(output_probs, 1)
        next_tokens = next_tokens.data.numpy()
        
        batched_input = torch.zeros((model.batch_size, token_num)).long()
        batched_input[np.arange(model.batch_size), next_tokens] = 1
        batched_input = Variable(batched_input.view(token_num, model.batch_size))
        
        program_probs *= top_probs.data.numpy()
        
        #prediction = map("".join,zip(prediction, char[]))
        for i in xrange(model.batch_size):
            prediction[i] += char[next_tokens[i]]
        
    return prediction, program_probs
