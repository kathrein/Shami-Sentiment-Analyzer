#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:59:14 2019

@author: xabuka
"""
from keras.datasets import imdb
from keras.preprocessing import sequence
import loading

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
#print('Loading data...')
#(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
#print(len(input_train), 'train sequences')
#print(len(input_test), 'test sequences')
#print('Pad sequences (samples x time)')
#input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
#input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
#print('input_train shape:', input_train.shape)
#print('input_test shape:', input_test.shape)

input_train, y_train, x_val,y_val = loading.load_train()
input_test, y_test = loading.load_test()
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from keras.models import Sequential
from keras.layers import Dense,SimpleRNN, Embedding

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(3, activation='sigmoid'))
model.summary()
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(x_val, y_val))
                    #validation_split=0.2)





import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

scores= model.evaluate(input_test, y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))