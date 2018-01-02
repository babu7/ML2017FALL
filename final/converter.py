#!/usr/bin/env python3
# ../../../fastText/fasttext print-word-vectors wiki.zh.bin < train.caption > train-fasttexted-byword.txt

# import sys.argv as argv
from sys import argv
import numpy as np
import pickle

if len(argv) < 4:
    print('./converter.py <caption> <fasttext-out> <list of sentence>')
    exit(-1)

cap = open(argv[1]).read()
cap_lines = cap.rstrip().split('\n')
cnt = [len(line.rstrip().split()) for line in cap_lines]

ctx = open(argv[2]).read()
lines = ctx.rstrip().split('\n')
lines = [ line.rstrip().split()[1:] for line in lines ]
lines = np.array(lines).astype(float)

s = 0
lines_words = []
for c in cnt:
    lines_words.append(lines[s:s+c,:])
    s += c

def sp_test(n_c):
    # [choice1, 2, 3, 4, n=2 choice1, 2, 3, 4...]
    z = [n_c[i:i+4] for i in range(0, len(n_c), 4)]
    # Transpose
    z = [[z[j][i] for j in range(len(z))] for i in range(4)]
    return z

lines_words = sp_test(lines_words)
pickle.dump(lines_words, open(argv[3], 'wb'))
