from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# train in PAL senti
# just building NN without any LSTM or CNN
#train_directory
imdb_dir = '../splitedPalSent'

train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
#i = 0
for label_type in ['pos','neg','no']:#categories
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] != 'tore':
            f = open(os.path.join(dir_name, fname))
            #i += 1
            #print(fname) 
            texts.append(f.read())# add the sentences to text array
            f.close()
            if label_type == 'pos':# which value to assign to every class
                labels.append(1)
            elif label_type == 'neg':# which value to assign to every class
                labels.append(0)
            elif label_type == 'no':# which value to assign to every class
                labels.append(2)
            








maxlen = 100  # We will cut reviews after 100 words
training_samples = 700  # We will be training on 200 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import LSTM


embedding_dim = 100

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())  # withuoot using LSTM it is better
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(x_val, y_val))


model.save_weights('Pal_senti_model.h5')

#prepare test set
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
#i = 0
for label_type in ['pos','neg','no']:#categories
    dir_name = os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] != 'tore':
            f = open(os.path.join(dir_name, fname))
            #i += 1
            #print(fname) 
            texts.append(f.read())# add the sentences to text array
            f.close()
            if label_type == 'pos':# which value to assign to every class
                labels.append(1)
            elif label_type == 'neg':# which value to assign to every class
                labels.append(0)
            elif label_type == 'no':# which value to assign to every class
                labels.append(2)
            

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
y_test = to_categorical(y_test, num_classes=3)

model.load_weights('Pal_senti_model.h5')
accuracy = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (accuracy[1]*100))