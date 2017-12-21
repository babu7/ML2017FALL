#!/usr/bin/env python3
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Flatten, Dot, Add, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import os
import argparse
from utils import load_train, load_test, DataSet, loadpkl

d_train_data = 'input_data/train.csv'
d_test_data  = 'input_data/test.csv'
d_epochs = 3
d_batch_size = 32

def get_model(n_users, n_items, latent_dim=6666):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adam')
    return model

def nn_model(n_users, n_items, lattent_dim=7777):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, lattent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, lattent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dense(150, activation='relu')(merge_vec)
    hidden = Dense(50, activation='relu')(hidden)
    output = Dense(1)(hidden)
    model = Model([user_input, item_input], output)
    model.compile(loss='mse', optimizer='sgd')
    return model

# Get embedding
def get_emb(model):
    user_emb = np.array(model.layers[2].get_weights()).squeeze()
    print('user embedding shape: ' % str(user_emb.shape))
    movie_emb = np.array(model.layers[3].get_weights()).squeeze()
    print('movie embedding shape: ' % str(movie_emb.shape))
    np.save('user_emb.npy', user_emb)
    np.save('movie_emb.npy', movie_emb)

def main(workdir='', action='train', modelpath=None, train_data=d_train_data,
        test_data=d_test_data, predict=None, epochs=d_epochs, batch_size=d_batch_size, prev=None):
    if modelpath is None:
        modelpath = os.path.join(workdir, 'model.h5')
    if predict is None:
        predict = os.path.join(workdir, 'predict.csv')
    if action == 'train':
        uid, mid, rid = load_train(train_data)
        d = DataSet(uid=uid, mid=mid, rid=rid)
        val = d.split(0, 0.2)
        d.shuffle()
        model = get_model(6041, 3953, 512)
        print(model.summary())
        d.save(workdir, 'train.pkl')
        val.save(workdir, 'val.pkl')
        tbcallback = TensorBoard(log_dir=os.path.join(workdir, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint(filepath=modelpath,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_loss')
        model.fit([d.uid, d.mid], d.rid,
                validation_data=([val.uid, val.mid], val.rid),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint, tbcallback, earlystopping]
                )
    elif action == 'test':
        model = load_model(modelpath)
        uid, mid = load_test(test_data)
        y = model.predict([uid, mid])
        y = y.flatten().clip(0, 5)
        with open(predict, 'w') as f:
            print('TestDataID,Rating', file=f)
            for k, v in enumerate(y):
                print("%d,%.0f" % (k+1, v), file=f)
    elif action == 'train2':
        prev_model = os.path.join('models', prev, 'model.h5')
        d = loadpkl('models', prev, 'train.pkl')
        val = loadpkl('models', prev, 'val.pkl')
        d.save(workdir, 'train.pkl')
        val.save(workdir, 'val.pkl')
        model = load_model(prev_model)
        tbcallback = TensorBoard(log_dir=os.path.join(workdir, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint(filepath=modelpath,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_loss')
        model.fit([d.uid, d.mid], d.rid,
                validation_data=([val.uid, val.mid], val.rid),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint, tbcallback, earlystopping]
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('action', choices=['train','test', 'train2'])
    parser.add_argument('--load_model', default = None)
    parser.add_argument('--train_data', default = d_train_data)
    parser.add_argument('--epochs', default = d_epochs, type=int)
    parser.add_argument('--batch_size', default = d_batch_size, type=int)
    parser.add_argument('--test_data', default = d_test_data)
    parser.add_argument('--predict')
    parser.add_argument('--previous')
    args = parser.parse_args()
    workdir = os.path.join('models', args.model_name)
    if not os.path.isdir(workdir) and args.action == 'train':
        os.makedirs(workdir)
    main(workdir, args.action, args.load_model, args.train_data, args.test_data,
            args.predict, args.epochs, args.batch_size, args.previous)
