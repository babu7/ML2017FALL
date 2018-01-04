#!/usr/bin/env python3
import os, sys
import importlib
import argparse
import numpy as np
import pickle
from shutil import copyfile

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from utils import DataSet, load_img, load_test_ls, get_test

multi_tsne = True
if multi_tsne:
    from MulticoreTSNE import MulticoreTSNE as TSNE
else:
    from sklearn.manifold import TSNE

def cnn():
    input_img = Input(shape=(784,))  # adapt this if using `channels_first` image data format

    x = Reshape((28, 28, 1))(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Reshape((784,))(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)
    return autoencoder, encoder

def dnn_simple():
    input_img = Input(shape=(784,))
    encoded = Dense(32, activation='relu')(input_img)

    decoded = Dense(784, activation='sigmoid')(encoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    return autoencoder, encoder

def dnn():
    input_img = Input(shape=(784,))
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    return autoencoder, encoder

def denoisy_cnn():
    input_img = Input(shape=(784,))  # adapt this if using `channels_first` image data format

    x = Reshape((28, 28, 1))(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Reshape((784,))(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)
    return autoencoder, encoder

def get_model():
    return dnn()

def main(args):
    workdir = os.path.join('models', args.model)
    modelpath = os.path.join(workdir, 'model.h5')
    model_arch = os.path.join(workdir, 'model_arch.py')
    action = args.action
    if action == 'train':
        # denoiser = load_model('models/denoise-cnn/model.h5')
        autoencoder, encoder = get_model()
        print(autoencoder.summary())

        x_train = load_img()
        # x_train = denoiser.predict(x_train)
        # noise_factor = 0.5
        # x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        # x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        copyfile(sys.argv[0], model_arch)

        tbcallback = TensorBoard(log_dir=os.path.join(workdir, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint(filepath=modelpath,
                                     verbose=1,
                                     monitor='val_loss',
                                     save_best_only=True)
        autoencoder.fit(x_train, x_train,
                validation_split=0.1,
                epochs=1024,
                batch_size=128,
                shuffle=True,
                # validation_data=(x_test, x_test),
                callbacks=[tbcallback, earlystopping, checkpoint])

    if action == 'train2':
        autoencoder, encoder = get_model()
        autoencoder.load_weights(args.load_model)
        x_train = load_img()
        copyfile(sys.argv[0], model_arch)

        tbcallback = TensorBoard(log_dir=os.path.join(workdir, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=40)
        checkpoint = ModelCheckpoint(filepath=modelpath,
                                     verbose=1,
                                     monitor='val_loss',
                                     save_best_only=True)
        autoencoder.fit(x_train, x_train,
                validation_split=0.1,
                epochs=1024,
                batch_size=128,
                shuffle=True,
                # validation_data=(x_test, x_test),
                callbacks=[tbcallback, earlystopping, checkpoint])

    if action == 'tsne':
        print('Loading ...')
        img = load_img()
        copyfile(sys.argv[0], model_arch)
        print('TSNE ...')
        if multi_tsne:
            img_embedded = TSNE(n_components=16, verbose=1, n_iter=4000, n_jobs=8).fit_transform(img)
        else:
            img_embedded = TSNE(n_components=16, verbose=1).fit_transform(img)
        pickle.dump(img_embedded, open(os.path.join(workdir, 'tsned_embed.pkl'), 'wb'))
        print('Clustering ...')
        kmeans = KMeans(n_clusters=2, n_init=20, max_iter=500, n_jobs=8).fit(img_embedded)
        print('Predicting ...')
        y = kmeans.labels_
        # Because waiting TSNE is too long, so save them
        pickle.dump(y, open(os.path.join(workdir, 'tsned_labels.pkl'), 'wb'))
        y = get_test(y, load_test_ls())
        with open(os.path.join(workdir, 'cluster.txt'), 'w') as f:
            print('ID,dataset,dataset', file=f)
            for i in range(y.shape[0] // 2):
                print("%d,%d,%d" % (i, y[i*2], y[i*2+1]), file=f)
        with open(os.path.join(workdir, 'predict.csv'), 'w') as f:
            print("ID,Ans", file=f)
            for i in range(y.shape[0] // 2):
                if y[i*2] == y[i*2+1]:
                    print("%d,1" % i, file=f)
                else:
                    print("%d,0" % i, file=f)

    if action == 'test':
        m, encoder = importlib.import_module(model_arch.rstrip('.py').replace('/','.')).get_model()
        print('Loading data ...')
        m.load_weights(modelpath)
        img = load_img()
        print("img shape: %s" % str(img.shape))
        print('Encoding ...')
        img_encoded = encoder.predict(img)
        img_encoded = img_encoded.reshape(img_encoded.shape[0], -1)
        # print('TSNE ...')
        # if multi_tsne:
        #     img_embedded = TSNE(n_components=2, verbose=1, n_jobs=8).fit_transform(img_encoded)
        # else:
        #     img_embedded = TSNE(n_components=2, verbose=1).fit_transform(img_encoded)
        print('Clustering ...')
        kmeans = KMeans(n_clusters=2, n_init=20, max_iter=500, n_jobs=8).fit(img_encoded)
        print('Predicting ...')
        y = kmeans.labels_

        # Because TSNE is too long, so save them
        # pickle.dump(y, open(os.path.join(workdir, 'tsned_labels.pkl'), 'wb'))
        y = get_test(y, load_test_ls())
        with open(os.path.join(workdir, 'cluster.txt'), 'w') as f:
            print('ID,dataset,dataset', file=f)
            for i in range(y.shape[0] // 2):
                print("%d,%d,%d" % (i, y[i*2], y[i*2+1]), file=f)
        with open(os.path.join(workdir, 'predict.csv'), 'w') as f:
            print("ID,Ans", file=f)
            for i in range(y.shape[0] // 2):
                if y[i*2] == y[i*2+1]:
                    print("%d,1" % i, file=f)
                else:
                    print("%d,0" % i, file=f)

    if action == 'plot':
        autoencoder, encoder = importlib.import_module(model_arch.rstrip('.py').replace('/','.')).get_model()
        autoencoder.load_weights(modelpath)
        img = load_img()
        x_train = get_test(img, load_test_ls())[:100]
        # noise_factor = 0.5
        # x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        # x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 4))
        decoded_imgs = autoencoder.predict(x_train)
        n = 10
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i+1)
            plt.imshow(x_train[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n+1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig = plt.gcf()
        fig.savefig(os.path.join(workdir, 'tmp.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('action', choices=['train','test','plot', 'tsne', 'train2'])
    parser.add_argument('--load_model', default = None)
    parser.add_argument('--train_data', default = None)
    parser.add_argument('--epochs', default = 3, type=int)
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--test_data', default = None)
    parser.add_argument('--predict')
    args = parser.parse_args()
    workdir = os.path.join('models', args.model)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    main(args)
