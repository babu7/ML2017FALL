#!/usr/bin/env python3
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Flatten, Dot, Add, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import os
import argparse
from utils import load_train, load_test

d_train_data = 'input_data/train.csv'
d_test_data  = 'input_data/test.csv'

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

def load_test(filename='input_data/test.csv'):
    lines = open(filename).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split(',') for line in lines]
    lines = np.array(lines).astype('int')
    user_id, movie_id = lines[:,1:2], lines[:,2:3]
    return user_id, movie_id

def load_train(filename='input_data/train.csv'):
    lines = open(filename).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split(',') for line in lines]
    lines = np.array(lines).astype('int')
    user_id , movie_id, rating_id = lines[:,1:2], lines[:,2:3], lines[:,3:4]
    return user_id, movie_id, rating_id

def main(workdir='', action='train', modelpath=None, train_data=d_train_data, test_data=d_test_data, predict=None):
    if modelpath is None:
        modelpath = os.path.join(workdir, 'model.h5')
    if predict is None:
        predict = os.path.join(workdir, 'predict.csv')
    if action == 'train':
        uid, mid, rid = load_train(train_data)
#         per = np.random.permutation(len(uid))
#         uid = uid[per]
#         mid = mid[per]
#         rid = rid[per]
        # users in train.csv:   6040
        # users in users.csv:   6040
        # movies in train.csv:  3688
        # movies in movie.csv:  3883
        model = get_model(6041, 3689, 1024)
        print(model.summary())
        tbcallback = TensorBoard(log_dir=os.path.join(workdir, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint(filepath=modelpath,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_loss')
        model.fit([uid, mid], rid, epochs=3, validation_split=0.2,
                callbacks=[checkpoint, tbcallback, earlystopping]
                )
    elif action == 'test':
        model = load_model(modelpath)
        uid, mid = load_test(test_data)
        y = model.predict([uid, mid])
        y = y.flatten().clip(0, 5)
        print(y.shape)
        with open(predict, 'w') as f:
            print('TestDataID,Rating', file=f)
            for k, v in enumerate(y):
                print("%d,%.0f" % (k, v), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('action', choices=['train','test'])
    parser.add_argument('--load_model', default = None)
    parser.add_argument('--train_data', default = d_train_data)
    parser.add_argument('--test_data', default = d_test_data)
    parser.add_argument('--predict')
    args = parser.parse_args()
    workdir = os.path.join('models', args.model_name)
    if not os.path.isdir(workdir) and args.action == 'train':
        os.makedirs(workdir)
    main(workdir, args.action, args.load_model, args.train_data, args.test_data, args.predict)
