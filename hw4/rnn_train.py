#!/usr/bin/env python3
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from keras.models import load_model

import numpy as np
import argparse
import pickle
import utils as u

train_label_x = 'input_data/trimmed_inputs.txt'
train_label_y = 'input_data/trimmed_labels.txt'
train_nolabel = 'input_data/training_nolabel.txt'

def collect(model, d, batch_size):
    pred = model.predict(d.inputs, batch_size=batch_size)
    mask = np.logical_or(pred>0.95, pred<0.05).flatten()
    coll = d.inputs[mask]
    pred = np.rint(pred[mask]).flatten()
    mask = np.logical_not(mask)
    d.inputs = d.inputs[mask]
    newdata = u.DataSet(coll, pred)
    return newdata

def main(model_name: str='model', load_pkl: bool=False, save_pkl: bool=False,
        fast: bool=False, num_words: int=20000, maxlen: int=80, batch_size: int=32,
        reinforce_step: int=0, **kwargs):
    print('Loading data...')
    if not load_pkl:
        d1, wdict = u.load_data(train_label_x, train_label_y, num_words=num_words)
        with open(model_name + '_wdict.pkl', 'wb') as f:
            pickle.dump(wdict, f)
        d2, wdict = u.load_data(train_nolabel, wdict=wdict)
        d1_valid = d1.split(0.2)
        d1.inputs = sequence.pad_sequences(d1.inputs, maxlen=maxlen)
        d1_valid.inputs = sequence.pad_sequences(d1_valid.inputs, maxlen=maxlen)
        d2.inputs = sequence.pad_sequences(d2.inputs, maxlen=maxlen)
        if save_pkl:
            with open('input_data/quick.pkl', 'wb') as f:
                pickle.dump((d1, d1_valid, d2), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('input_data/quick.pkl', 'rb') as f:
            d1, d1_valid, d2 = pickle.load(f)

    if fast:
        d1.shrink(0.001)
        d1_valid.shrink(0.001)
        d2.shrink(0.001)

    print('Pad sequences (samples x time)')
    print('x_train shape: %s' % str(d1.inputs.shape))
    print('x_test  shape: %s' % str(d1_valid.inputs.shape))

    model = Sequential()
    model.add(Embedding(num_words, 128))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_acc', patience=5)
    checkpointer = ModelCheckpoint(filepath=model_name+'_weight.h5', monitor='val_acc', verbose=1, save_best_only=True)

    model.fit(d1.inputs, d1.labels,
              batch_size=batch_size,
              epochs=5,
              validation_data=(d1_valid.inputs, d1_valid.labels),
              callbacks=[earlystopping, checkpointer])

    # reinforce
    for i in range(reinforce_step):
        print("current step: %d" % i)
        newdata = collect(model, d2, batch_size)
        d1.join(newdata)
        model.fit(d1.inputs, d1.labels,
                  batch_size=batch_size,
                  epochs=2,
                  validation_data=(d1_valid.inputs, d1_valid.labels),
                  callbacks=[earlystopping, checkpointer])

    score, acc = model.evaluate(d1_valid.inputs, d1_valid.labels, batch_size=batch_size)
    print('Valid loss: %f'% score)
    print('Valid accuracy: %f'% acc)

    if not load_pkl:
        test, wdict = u.load_data('input_data/trimmed_test.txt', wdict=wdict)
        test.inputs = sequence.pad_sequences(test.inputs, maxlen=maxlen)
        if save_pkl:
            with open('input_data/test.pkl', 'wb') as f:
                pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('input_data/test.pkl', 'rb') as f:
            test = pickle.load(f)

    print('test shape: %s' % str(test.inputs.shape))
    print('Predicting...')
    model = load_model(model_name+'_weight.h5')
    prob = np.rint(model.predict(test.inputs)).flatten()
    with open(model_name+'_predict.txt', 'w') as f:
        print('id,label', file=f)
        for i in range(prob.shape[0]):
            print("%d,%.0f" % (i, prob[i]), file=f)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('model_name')
    parser.add_argument("--load-pickle", action='store_true', dest='load_pkl')
    parser.add_argument("--save-pickle", action='store_true', dest='save_pkl')
    parser.add_argument("--fast", action='store_true',
            help="Enable fast mode to load partial data only")
    parser.add_argument('--max-features', type=int)
    parser.add_argument('--maxlen', type=int, help='cut texts after this number of words (among top num_words most common words)')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--reinforce_step', type=int)
    args = vars(parser.parse_args())
    main(**args)
