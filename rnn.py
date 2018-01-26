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
    'S' : 8
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

def program_to_input(program):
    return np.vectorize(lambda s: 'S' + s[:-1])(program)

def program_to_tensor(program):
    return torch.LongTensor([token[c] for c in program])

def token_to_tensor(input_token):
    return torch.LongTensor([token[input_token]])

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
                       # parameters for PQT + PG
                       learning_rate=1e-4,       # .\(ER) from paper
                       PQT_loss_multiplier=50.0,     # .\(TOPK) from paper
                       entropy_regularizer=0.01,     # .\(ENT) from paper
                       K = 10):
        super(BFgen, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.PQT_loss_multiplier = PQT_loss_multiplier
        self.entropy_regularizer = entropy_regularizer
        self.K = 10
        
        self.encoder = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.functional.log_softmax
        
        self.baseline = 0.
        self.entropy = Variable(torch.FloatTensor([0]))

        self.pqt_programs = np.array([])
        self.pqt_rewards = np.array([])

    """
    initialize weights and biases of the network
    """
    def init_weights(self, range_=0.1, factor=0.5, bias=1.0):
        # encoder + decoder
        self.encoder.weight.data.uniform_(-range_, range_)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-range_, range_)
        # lstm
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, bias) 
            else :
                nn.init.kaiming_normal(param, factor)               
        
        


    """
    forward
    Takes input_token and hidden memory state <- input to recursive layer
    returns output token and changed hidden memory state.
    """
    # input_token : length x batch
    def forward(self, input_token):
        embeds = self.encoder(input_token) # length x batch x token_size
        output, self.hidden = self.lstm(embeds, self.hidden) # length x batch x nhid
        decoded = self.decoder(output) # length x batch x output_size
        probs = self.logsoftmax(decoded.permute(2, 0, 1), dim=0) # output_size x length x batch
        return probs # output_size x length x batch
    
    def init_hidden_zero(self, batch_size):
        self.hidden = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),)
    
    def init_hidden_normal(self, batch_size, factor=0.5):
        self.hidden = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))) 
        nn.init.kaiming_normal(self.hidden[0], factor)
        nn.init.kaiming_normal(self.hidden[1], factor)
        # means = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        # std = torch.Tensor([variance]*self.hidden_size*self.n_layers*batch_size).unsqueeze(0)
        # self.hidden = (Variable(torch.normal(means, std)), Variable(torch.normal(means, std)))

    def save_best(self, rewards, programs):
        self.pqt_programs = np.append(self.pqt_programs, programs)
        self.pqt_rewards = np.append(self.pqt_rewards, rewards)
        args = np.argsort(self.pqt_rewards)
        self.pqt_programs = self.pqt_programs[args][-self.K:]
        self.pqt_rewards = self.pqt_rewards[args][-self.K:] # K

    def clear_pq(self):
        self.pqt_programs = np.array([])
        self.pqt_rewards = np.array([])

    def evaluate(self, batched_input):
        output_log_probs = self.forward(batched_input)
        self.entropy += -(torch.sum(output_log_probs * torch.exp(output_log_probs)))
        return output_log_probs

    # generate 1 program
    def sample(self, predict_len=100):
        input_token = 'S'
        input_token = token_to_tensor(input_token).view(1, 1)
        input_token = Variable(input_token)

        self.init_hidden_normal(1, factor=0.5)
        prediction = ""

        for i in range(predict_len):
            output_probs = self.forward(input_token)

            
            top_probs, input_token = torch.max(output_probs, 0)

            prediction += char[input_token.data[0][0]]

        return prediction

