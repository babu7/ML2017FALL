#!/usr/bin/env python3
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import utils as u

train_label_x = 'input_data/trimmed_inputs.txt'
train_label_y = 'input_data/trimmed_labels.txt'
train_nolabel = 'input_data/training_nolabel.txt'

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def collect(model, d):
    pred = model.predict(d.inputs, batch_size=batch_size)
    mask = np.logical_or(pred>0.9, pred<0.1).flatten()
    coll = d.inputs[mask]
    pred = np.rint(pred[mask]).flatten()
    mask = np.logical_not(mask)
    d.inputs = d.inputs[mask]
    newdata = u.DataSet(coll, pred)
    return newdata

def main():
    print('Loading data...')
    d1, wdict = u.load_data(train_label_x, train_label_y, num_words=10000)
    d2, wdict = u.load_data(train_nolabel, wdict=wdict)
    d1_valid = d1.split(0.3)

    d1.shrink(0.001)
    d1_valid.shrink(0.001)
    d2.shrink(0.001)

    print('Pad sequences (samples x time)')
    d1.inputs = sequence.pad_sequences(d1.inputs, maxlen=maxlen)
    d1_valid.inputs = sequence.pad_sequences(d1_valid.inputs, maxlen=maxlen)
    d2.inputs = sequence.pad_sequences(d2.inputs, maxlen=maxlen)
    print('x_train shape: %s' % str(d1.inputs.shape))
    print('x_test  shape: %s' % str(d1_valid.inputs.shape))

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(d1.inputs, d1.labels,
              batch_size=batch_size,
              epochs=3,
              validation_data=(d1_valid.inputs, d1_valid.labels))

    # reinforce
    step = 5
    for i in range(step):
        print("current step: %d" % step)
        newdata = collect(model, d2)
        d1.join(newdata)
        model.fit(d1.inputs, d1.labels,
                  batch_size=batch_size,
                  epochs=3,
                  validation_data=(d1_valid.inputs, d1_valid.labels))

    score, acc = model.evaluate(d1_valid.inputs, d1_valid.labels, batch_size=batch_size)
    print('Test score: %f'% score)
    print('Test accuracy: %f'% acc)
    return model

if __name__ == '__main__':
    main()
