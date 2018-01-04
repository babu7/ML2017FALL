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

def load_img(filename='input_data/image.npy'):
    return np.load(filename).astype('float32') / 255

def load_test_ls(filename='input_data/test_case.csv', fast=False):
    # if fast:
    #     return np.load('input_data/test_case.npy')
    lines = open(filename).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split(',')[1:] for line in lines]
    lines = np.array(lines).astype('int')
    idx_ls = lines.flatten()
    return idx_ls
    img = load_img()
    ret = []
    for i in range(idx_ls.shape[0]):
        ret.append(img[idx_ls[i]])
    ret = np.array(ret)
    # np.save('input_data/test_case.npy', ret)
    return ret

def get_test(img, idx_ls):
    ret = []
    for i in range(idx_ls.shape[0]):
        ret.append(img[idx_ls[i]])
    ret = np.array(ret)
    # np.save('input_data/test_case.npy', ret)
    return ret

def loadpkl(*path):
    return pickle.load(open(os.path.join(*path), 'rb'))
