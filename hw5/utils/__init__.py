#!/usr/bin/env python3
import re
import numpy as np

class DataSet:
    def __init__(self, inputs, labels=None):
        self.inputs = inputs
        self.labels = labels
    def shuffle():
        permutation = np.random.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[permutation]
        self.labels = self.labels[permutation]
    def split(self, valid_rate):
        p = int(self.inputs.shape[0] * valid_rate)
        valid = DataSet(self.inputs[:p])
        self.inputs = self.inputs[p:]
        if self.labels is not None:
            valid.labels = self.labels[:p]
            self.labels = self.labels[p:]
        return valid
    def join(self, d):
        if d.inputs.size == 0:
            return
        self.inputs = np.concatenate((self.inputs, d.inputs), axis=0)
        if self.labels is not None:
            self.labels = np.concatenate((self.labels, d.labels), axis=0)
    def shrink(self, rate):
        n_s = int(self.inputs.shape[0] * rate)
        self.inputs = self.inputs[:n_s]
        if self.labels is not None:
            self.labels = self.labels[:n_s]

def load_train(filename):
    lines = open(filename).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split(',') for line in lines]
    lines = np.array(lines).astype('int')
    user_id , movie_id, rating_id = lines[:,1:2], lines[:,2:3], lines[:,3:4]
    return user_id, movie_id, rating_id

def load_test(filename):
    lines = open(filename).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split(',') for line in lines]
    lines = np.array(lines).astype('int')
    user_id, movie_id = lines[:,1:2], lines[:,2:3]
    return user_id, movie_id
