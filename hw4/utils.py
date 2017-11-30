#!/usr/bin/env python3
import sys
import argparse
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

def str2ls(fname):
    s = ''
    with open(fname, 'r') as f:
        s = f.read()
    s = re.sub(r'([0-9])([a-z])', r'\1 \2', s)
    s = re.sub(r'([a-z])([0-9])', r'\1 \2', s)
    s = s.splitlines()
    datals = [re.findall(r"[\w']+", line) for line in s]
    return datals

def word2id(datals, n_word=1000, wdict=None):
    if not wdict:
        flatten = [item for subls in datals for item in subls]

        # sort word by frequency
        wdict_full = {}
        for word in flatten:
            if word in wdict_full.keys():
                wdict_full[word] += 1
            else:
                wdict_full[word] = 1
        rankls = sorted(wdict_full.items(), key=lambda k: k[1], reverse=True)

        wdict= {}
        for i in range(min(n_word, len(wdict_full))):
            wdict[rankls[i][0]] = i+1

    datals = [[wdict[word] if word in wdict.keys() else 0 for word in line] for line in datals]
    return datals, wdict

def load_data(raw_x, raw_y=None, wdict=None):
    if not raw_y and not wdict:
        return -1
    data_x = str2ls(raw_x)
    data_x, wdict = word2id(data_x, 1000)
    data_x = np.array(data_x)
    data_y = None
    if raw_y:
        with open(raw_y, 'r') as f:
            data_y = f.read().splitlines()
            data_y = [int(i) for i in data_y]
            data_y = np.array(data_y)
    return DataSet(data_x, data_y), wdict

def loadall():
    train_label, wdict = load_data('input_data/trimmed_inputs.txt', 'input_data/trimmed_labels.txt')
    train_nolabel, wdict = load_data('input_data/training_nolabel.txt', wdict=wdict)
    return train_label, train_nolabel

def main():
    parser = argparse.ArgumentParser('Some utils')
    parser.add_argument('--count-word', type=str, dest='cntwd')
    parser.add_argument('--str2ls', type=str, dest='str2ls')
    opts = parser.parse_args()

    if opts.cntwd:
        with open(opts.cntwd, "r") as f:
            wdict = {}
            raw = re.findall(r"([a-z]+(?: ' )?[a-z]+)", f.read())
            for word in raw:
                if word in wdict.keys():
                    wdict[word] += 1
                else:
                    wdict[word] = 1

            wlist = sorted(wdict.items(), key=lambda k: k[1], reverse=True)
            print(wlist[:100])

    if opts.str2ls:
        str2ls(opts.str2ls)

if __name__ == '__main__':
    main()
