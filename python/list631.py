#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:59:14 2019

@author: xabuka
using RNN, LSTM, GRU , BI 
"""

import loading

max_features = 10000
max_len = 100
training_samples = 1000  # We will be training on 200 samples
validation_samples = 2000  # We will be validating on 10000 samples
data_dir = '../data/SplitedPalSent'
#'/Users/xabuka/PycharmProjects/measuring_acceptability/python-files/aclImdb' #

input_train, y_train = loading.load_train(data_dir,max_len,training_samples,validation_samples,max_features, Validation = False )
input_test, y_test = loading.load_test(data_dir,max_len,max_features)


print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from keras.models import Sequential
from keras.layers import Dense, Embedding,GRU,LSTM

model = Sequential()
model.add(Embedding(max_features, 64, input_length= max_len))
model.add(LSTM(32)) #GRU
#bidirectional 
#model.add(layers.Bidirectional(layers.LSTM(32)))
#model.add(layers.Bidirectional(
#    layers.GRU(32), input_shape=(None, float_data.shape[-1])))

#model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.summary()
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
batch_size = 32
history = model.fit(input_train, y_train,
                    epochs=9,
                    batch_size=batch_size,
                    #validation_data=(x_val, y_val),callbacks=[early_stopping])
                    validation_split=0.2)





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

from sklearn import metrics
import numpy as np
from keras.utils import to_categorical


scores= model.evaluate(input_test, y_test,verbose=0)
#yhat = model.predict_classes(input_test, verbose = 2, batch_size = batch_size)
yhat = model.predict(input_test, verbose = 2, batch_size = batch_size)
#print(type(yhat))
from numpy import array
yhat = array(loading.one_encode_pred(yhat))
#print(type(yhat))
#print(yhat)
#print(yhat.shape)
#print(y_test.shape)
#n_values = 3 
#c = np.eye(n_values, dtype=int)[np.argmax(yhat, axis=1)]

#yhat = to_categorical(yhat, num_classes=3)
#print(yhat)
print(metrics.classification_report(y_test[:,:], np.round(yhat[:,:])))
#print(np.round(yhat[:,:]))
#yhat = to_categorical(yhat[:,], num_classes=3)
#for x,y in zip(y_test[:,:],yhat[:,:]):
#    print(x,y)
#print(x,y )

print("Accuracy: %.2f%%" % (scores[1]*100))


#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#import array
#
#bad = "مش حلو بالمرة مقرف"
#good = "كان كتابا رائعا"
#tokenizer = Tokenizer(num_words=10000)
#tokenizer.fit_on_texts([good,bad])
#sequences = tokenizer.texts_to_sequences([good,bad])
#word_index = tokenizer.word_index
#
##predict sentiment from reviews
#
#for review in [good,bad]:
#    tmp = []
#    for word in review.split(" "):
#        tmp.append(tokenizer.texts_to_sequences(word))
#    tmp_padded = pad_sequences(tmp, maxlen=100) 
#    print("%s. Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))
#
