#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:59:14 2019

@author: xabuka
using CNN
"""

import loading

max_features = 10000  # number of words to consider as features
max_len  = 500
training_samples = 700  # We will be training on 200 samples
validation_samples = 200  # We will be validating on 10000 samples



x_train, y_train, x_val,y_val = loading.load_train(max_len,training_samples,validation_samples,max_features )
x_test, y_test = loading.load_test(max_len,max_features)
print('input_train shape:', x_train.shape)
print('input_test shape:', x_test.shape)


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
#from keras.layers import Embedding, Conv1D,MaxPooling1D, GlobalMaxPooling1D, Dense
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(3))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))




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

scores= model.evaluate(x_test, y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))