import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import bfCompiler
import rnn

def batch_reward(bf_inputs, bf_outputs, batch_size, B = 256, scaling_factor=0.1) :
    outputs_num = len(bf_outputs)
    def hamming_distance(seq1, seq2):
        assert len(seq1) == len(seq2)
        return sum(elem1 != elem2 for elem1, elem2 in zip(seq1, seq2))
    
    def d(program_output, expected_output):
        b_scalar = abs(len(program_output) - len(expected_output))
        (program_output, expected_output) = (program_output, expected_output[:len(program_output)])\
        if len(program_output) <= len(expected_output) else (program_output[:len(expected_output)], expected_output)
        return hamming_distance(program_output, expected_output) + B * b_scalar
    
    def S(program_output, expected_output):
        return d([], expected_output) - d(program_output, expected_output)
    
    def total_reward(program_code):
        program_outputs = map((lambda x: bfCompiler.BF(program_code, x)), bf_inputs)
        return scaling_factor * sum(S(program_output, bf_output) for program_output, bf_output in zip(program_outputs, bf_outputs))
    
    def reward_program(program_code_batch) :
        return np.array(map(lambda x: total_reward(x), program_code_batch))
        
    return reward_program

def objective_PG(model, reward_f, predict_len = 100, N = 1000):
    objective = Variable(torch.FloatTensor([1]))
    baseline = 0
    for i in range(0, N, model.batch_size):
        input_token = 'S'
        input_token = rnn.token_to_tensor(input_token)
        model.init_hidden_normal(model.batch_size, variance=0.5)
        prediction = [""] * model.batch_size
        policy_probs = Variable(torch.zeros(1, model.batch_size).float())

        batched_input = torch.zeros((model.batch_size, 1)).long()
        batched_input = batched_input + input_token
        batched_input = Variable(batched_input.view(1, model.batch_size))

        for i in range(predict_len):
            output_probs = model.evaluate(batched_input)
            
            top_probs, next_tokens = torch.max(output_probs, 0)
            batched_input = next_tokens
            next_tokens = next_tokens.view(model.batch_size)

            # accumulate objective
            policy_probs +=  torch.log(top_probs)
            
            # save chars
            for i in xrange(model.batch_size):
                prediction[i] += rnn.char[next_tokens[i].data[0]]
        
        # calculate rewards
        rewards = np.array(map(reward_f, prediction))
        rewards_mean = rewards.mean()
        model.baseline = rewards_mean * model.GAMMA + (1 - model.GAMMA) * model.baseline
        # exponential moving avarage
        # move to model, calculate over avarage of baselines
        rewards = rewards - model.baseline

        #save best for PQT
        model.save_best(rewards, prediction)
        
        rewards = Variable(torch.FloatTensor(rewards))
        objective = (rewards * policy_probs).sum()
    return objective / N,

def objective_PQT(model):
    objective = Variable(torch.FloatTensor([1]))
    model.init_hidden_normal(model.K)

    inputs = np.vectorize(rnn.program_to_input)(model.pqt_programs)
    inputs = torch.stack(np.vectorize(rnn.program_to_tensor)(inputs), dim=1)
    inputs = Variable(inputs) # 100 x 10
    outputs = torch.stack(np.vectorize(rnn.program_to_tensor)(model.pqt_programs), dim=0)
    outputs = Variable(outputs) # 10 x 100

    probs = model.forward(inputs) # 8 x 100 x 10
    probs = probs.permute(2, 1, 0) # 10 x 100 x 8
    probs = torch.gather(probs, 2, outputs.unsqueeze(2)).squeeze(2)

    objective = probs.log().sum()

    return objective / model.K
