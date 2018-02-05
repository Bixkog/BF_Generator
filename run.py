import rnn
import train
import brainfuck

embedding_size = 10
hidden_size = 35
output_size = len(rnn.char.keys())
n_layers = 2
batch_size = 64
GAMMA = 0.99 # for exponential moving avarage constant from paper





model = rnn.BFgen(rnn.token_num, 
		embedding_size, hidden_size, 
		output_size, n_layers, batch_size)

model.clear_pq()
model.init_weights()

def reverse(s):
    return s[::-1]

reverse_in, reverse_out, scaling_factor = train.gen_tests(reverse)
reverse_reward = train.batch_reward(reverse_in, 
	reverse_out, 
	scaling_factor=scaling_factor)

train.train_pqt_pg(model, 
	reverse_reward, NPE=20000000)
