#!/usr/bin/env python3
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

class DataSet:
    def __init__(self, inputs):
        self.inputs = inputs
    def shuffle():
        permutation = np.random.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[permutation]
        self.labels = self.labels[permutation]

def load_data():
    npz = np.load('input_data/train.npz')
    data = DataSet(npz['inputs'])
    data.labels = npz['labels']
    return data

def train_model(dataSet):
    # CNN
    model = Sequential()
    model.add(Conv2D(filters=25, kernel_size=(3, 3), input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Fully connected
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(dataSet.inputs, dataSet.labels, batch_size=100, epochs=20, validation_split=0.3)
#     # CNN
#     model = Sequential()
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(48, 48, 1)))
#     model.add(Conv2D(filters=32, kernel_size=(3, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.1))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.1))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(48, 48, 1)))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.4))
#     model.add(Flatten())
#     # Fully connected
#     model.add(Dense(2048, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#     model.fit(dataSet.inputs, dataSet.labels, batch_size=100, epochs=40, validation_split=0.3)
    return model

def main():
    face_data = load_data()

    model = train_model(face_data)

if __name__ == '__main__':
    main()
