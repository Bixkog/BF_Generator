import rnn
import train
import brainfuck

import argparse
import multiprocessing
import ctypes
import numpy as np

def reverse(s):
    return s[::-1]

bf_inputs, bf_outputs, B, scaling_factor, pool = 0, 0, 0, 0, 0

def hamming_distance(seq1, seq2):
    return sum(abs(ord(elem1) - ord(elem2)) for elem1, elem2 in zip(seq1, seq2))

def d(seq1, seq2):
    b_scalar = abs(len(seq1) - len(seq2))
    (seq1, seq2) = (seq1, seq2[:len(seq1)])\
    if len(seq1) <= len(seq2) else (seq1[:len(seq2)], seq2)
    return hamming_distance(seq1, seq2) + B.value * b_scalar

def S(program_output, expected_output):
    return  d([], expected_output) - d(program_output, expected_output)

def total_reward(program_code):
    program_outputs = map((lambda x: brainfuck.BF(program_code, x)), bf_inputs)
    # program_outputs = map(lambda (x,y,z):x, program_outputs)
    return scaling_factor.value * sum(S(program_output, bf_output) 
    			for program_output, bf_output in zip(program_outputs, bf_outputs))

# correct_reward = scaling_factor * sum(S(bf_output, bf_output) for bf_output in bf_outputs)
def reward_program(program_code_batch):
    rewards = np.array(pool.map(total_reward, program_code_batch))
    # max_reward = max(1e-8, correct_reward)
    # return np.vectorize(lambda x: max(-1, x))(rewards / max_reward)
    return rewards

def parallel_batch_reward(bf_inputs_, bf_outputs_, B_=256, scaling_factor_=0.1):
	global bf_inputs
	bf_inputs = multiprocessing.Array(ctypes.c_char_p, bf_inputs_)
	global bf_outputs
	bf_outputs = multiprocessing.Array(ctypes.c_char_p, bf_outputs_)
	global B
	B = multiprocessing.Value("i", B_)
	global scaling_factor
	scaling_factor = multiprocessing.Value("f", scaling_factor_)
	global pool
	pool = multiprocessing.Pool()
	return reward_program



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--load", help="load model from models dir,\
										base name needed")
	parser.add_argument("-n", "--name", help="model name for saving")
	parser.add_argument("-p", "--parallel", 
		help="use multiprocessing to interpret BF",
		action="store_true")

	args = parser.parse_args()

	embedding_size = 10
	hidden_size = 35
	output_size = len(rnn.char.keys())
	n_layers = 2
	batch_size = 64
	GAMMA = 0.99 # for exponential moving avarage constant from paper

	model = rnn.BFgen(rnn.token_num, 
			embedding_size, hidden_size, 
			output_size, n_layers, batch_size)

	if args.load:
		model.load(args.load)
	else:
		model.clear_pq()
		model.init_weights()

	if not args.name:
		args.name = "model"

	reverse_in, reverse_out, scaling_factor = train.gen_tests(reverse)
	
	if args.parallel:
		reverse_reward = parallel_batch_reward(reverse_in, 
			reverse_out, 
			scaling_factor_=scaling_factor)
	else:
		reverse_reward = train.batch_reward(reverse_in, 
			reverse_out, 
			scaling_factor=scaling_factor)

	train.train_pqt_pg(model, 
		reverse_reward, NPE=20000000, file_name=args.name)
