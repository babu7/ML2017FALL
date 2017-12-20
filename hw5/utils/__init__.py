#!/usr/bin/env python3
import os
import pickle
import re
import numpy as np

class DataSet:
    def __init__(self, **data):
        assert isinstance(data, dict)
        item = list(data.values())[0]
        assert isinstance(item, np.ndarray)
        self.__len__ = item.shape[0]
        self.__data__ = []
        for k, v in data.items():
            assert len(v) == self.__len__
            setattr(self, k, v)
            self.__data__.append(k)
    def shuffle(self):
        permutation = np.random.permutation(self.__len__)
        for name in self.__data__:
            setattr(self, name, getattr(self, name)[permutation])
    def split(self, begin, end=None):
        # begin is split rate when there's only one argument
        if end is None:
            end = begin
            begin = 0
        p1 = int(self.__len__ * begin)
        p2 = int(self.__len__ * end)
        dic = {}
        for name in self.__data__:
            dic[name] = getattr(self, name)[p1:p2]
            m = np.ones(self.__len__, dtype=bool)
            m[p1:p2] = False
            setattr(self, name, getattr(self, name)[m])
        self.__len__ = self.__len__ -(p2 - p1)
        return DataSet(**dic)
    def join(self, d):
        if d.__len__ == 0:
            return
        for name in self.__data__:
            con = np.concatenate((getattr(self, name), getattr(d, name)), axis=0)
            setattr(self, name, con)
        self.__len__ = self.__len__ + d.__len__
    def shrink(self, rate):
        end = int(self.__len__ * rate)
        for name in self.__data__:
            setattr(self, name, getattr(self, name)[:end])
        self.__len__ = end
    def save(self, *path):
        pickle.dump(self, open(os.path.join(*path), 'wb'))
    @property
    def len(self):
        return self.__len__
    @property
    def data(self):
        return self.__data__

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

def loadpkl(*path):
    return pickle.load(open(os.path.join(*path), 'rb'))
