#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:44:59 2019

@author: xabuka
 
from Book Deep learning with Keras
Page 186
list 6.21
الجزء الاول بدون كيراس
تشرح كيفية عمل ال RNN
الجزء الثاني نفس الشي ولكن باستخدام كيراس
"""
#import numpy as np
#
#timesteps = 100  # Number of timesteps in the input sequence
#inputs_features = 32  # Dimensionality of the input feature space
#output_features = 64  # Dimensionality of the output feature space
## This is our input data - just random noise for the sake of our example.
#inputs = np.random.random((timesteps, inputs_features))
## This is our "initial state": an all-zero vector.
#state_t = np.zeros((output_features,))
## Create random weight matrices
#W = np.random.random((inputs_features, output_features))
#U = np.random.random((output_features, output_features))
#b = np.random.random((output_features,))
#successive_outputs = []
#
#print(inputs.shape)
#print(state_t.shape)
#print(W.shape)
#print(U.shape)
#print(b.shape)
#
#
#for input_t in inputs:  # input_t is a vector of shape (input_features,)
#    # We combine the input with the current state
#    # (i.e. the previous output) to obtain the current output.
#    output_t = np.tanh(np.dot(input_t,W) + np.dot(U, state_t) + b)
#    # We store this output in a list.
#    successive_outputs.append(output_t)
#    # We update the "state" of the network for the next timestep
#    state_t = output_t
## The final output is a 2D tensor of shape (timesteps, output_features).
#final_output_sequence = np.concatenate(successive_outputs, axis=0)
#print((successive_outputs[0]))


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
#model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
