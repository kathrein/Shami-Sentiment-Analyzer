#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:38:01 2019

@author: xabuka
"""

import loading

max_features = 20000
max_len = 100
training_samples = 7500  # We will be training on 200 samples
validation_samples = 1000  # We will be validating on 10000 samples
data_dir = '../data/labr'

x_train, y_train, x_val,y_val = loading.load_train_binary(data_dir,max_len,training_samples,validation_samples,max_features )
x_test, y_test = loading.load_test_binary(data_dir,max_len,max_features)
print('input_train shape:', x_train.shape)
print('input_test shape:', x_test.shape)


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Embedding, Conv1D,MaxPooling1D, Dense,Dropout,LSTM




# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
epochs = 2

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1,activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, 
          epochs=epochs,shuffle=True, 
          validation_data=(x_val, y_val),callbacks=[early_stopping])



# testing part, in 2 class matrixes. 



from sklearn import metrics
import numpy as np
#print(y_val[:,:])


x_test, y_test = loading.load_test_binary(data_dir,max_len,max_features)
yhat = model.predict(x_test, verbose = 2, batch_size = batch_size)
print(metrics.classification_report(y_test[:], np.round(yhat[:]) ,target_names = ["negative", "positive"]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

score = ['negative', 'positive']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(score)))
    plt.xticks(tick_marks, score, rotation=45)
    plt.yticks(tick_marks, score)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = metrics.confusion_matrix(y_test[:], np.round(yhat[:]))
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


scores= model.evaluate(x_test, y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))