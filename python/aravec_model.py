#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:04:30 2019

@author: xabuka

I want to learn here how to import aravec and learn a model using it.
"""

import loading
import process_aravec
from gensim.models import Word2Vec
import os
import numpy as np
from keras.preprocessing.text import Tokenizer



max_features = 10000
max_len = 100
training_samples = 11759  # We will be training on 200 samples
validation_samples = 2000  # We will be validating on 10000 samples
data_dir = '../data/SplitedPalSent'#SplitedPalSent'
ara_dir = '../data/tweet_cbow_300/'

#'/Users/xabuka/PycharmProjects/measuring_acceptability/python-files/aclImdb' #

input_train, y_train = loading.load_train(data_dir,max_len,training_samples,validation_samples,max_features, Validation = False )
input_test, y_test = loading.load_test(data_dir,max_len,max_features)


word_index = loading.word_index(data_dir,max_len,max_features)

t_model = Word2Vec.load(os.path.join(ara_dir,'tweets_cbow_300'))


print('Found %s word vectors.' % len(t_model.wv.index2word))# how many words in aravec this model
embeddings_index = t_model.wv

embedding_dim = embeddings_index.vector_size #300
embedding_matrix = np.zeros((max_features, embedding_dim))

for word, i in word_index.items():
    word = process_aravec.clean_str(word).replace(" ", "_")
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
        if i < max_features:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                #print(len(embedding_vector))
                
#    else:
#        print(word)
            
#print(embedding_matrix)
#for x in embedding_matrix:
#    print(x)
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Embedding, Conv1D,MaxPooling1D, Dense,Dropout,LSTM



# Convolution
kernel_size = 5
filters = 64
pool_size = 2

# LSTM
lstm_output_size = 70


model = Sequential()
model.add(Embedding(max_features, 300,weights=[embedding_matrix], trainable=False, input_length=max_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(3,activation='sigmoid'))
print(model.summary())      


model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])

history = model.fit(input_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_split= 0.1)


model.save_weights('pre_trained_aravec_model.h5')               




#test
from sklearn import metrics
import numpy as np
                
#x_test, y_test = loading.load_test(data_dir,maxlen,max_features)
model.load_weights('pre_trained_aravec_model.h5')
#metrics_names= model.evaluate(input_test, y_test)
#print("{}: {}".format(model.metrics_names[0], metrics_names[0]))
#print("{}: {}".format(model.metrics_names[1], metrics_names[1]))
yhat = model.predict(input_test, verbose = 2, batch_size = 32)
print(metrics.classification_report(y_test.argmax(axis=1), yhat.argmax(axis=1)))



import matplotlib.pyplot as plt
import numpy as np

#score = ['negative', 'positive']
#
#def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greys):
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(set(score)))
#    plt.xticks(tick_marks, score, rotation=45)
#    plt.yticks(tick_marks, score)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    
## Compute confusion matrix
#cm = metrics.confusion_matrix(y_test[:,1], np.round(yhat[:,1]))
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cm)    
#
#cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#plt.figure()
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#
#plt.show()
from sklearn.metrics import accuracy_score
predicted = model.predict(input_test)
predicted = np.argmax(predicted, axis=1)
accuracy_score(y_test.argmax(axis=1), predicted)
print(accuracy_score(y_test.argmax(axis=1), predicted))

matrix = metrics.confusion_matrix(y_test.argmax(axis=1), yhat.argmax(axis=1))
print(matrix)


scores= model.evaluate(input_test, y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


import seaborn as sns             
conf_mat = metrics.confusion_matrix(y_test.argmax(axis=1), yhat.argmax(axis=1))
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=['negative','positive','no'], yticklabels=['negative','positive','no'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()