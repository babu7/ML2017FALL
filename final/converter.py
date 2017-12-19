#!/usr/bin/env python3

# import sys.argv as argv
from sys import argv
import numpy as np

if len(argv) < 4:
    print('./converter.py <caption> <fasttext-out> <emb.npy>')
    exit(-1)

ctx = open(argv[2]).read()
lines = ctx.rstrip().split('\n')
lines = [ line.rstrip().split()[1:] for line in lines ]
lines = np.array(lines).astype(float)

cap = open(argv[1]).read()
cap_lines = cap.rstrip().split('\n')
cnt = [len(line.rstrip().split()) for line in cap_lines]

s = 0
lines_words = []
for c in cnt:
    lines_words.append(lines[s:s+c,:])
    s += c

np.save(argv[3], lines_words)
