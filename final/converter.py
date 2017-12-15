#!/usr/bin/env python3

# import sys.argv as argv
from sys import argv
import numpy as np

if len(argv) < 3:
    print('./converter.py <fasttext-out> <emb.npy>')
    exit(-1)

ctx = open(argv[1]).read()
lines = ctx.rstrip().split('\n')
lines = [ line.rstrip().split()[1:] for line in lines ]
lines = np.array(lines).astype(float)
np.save(argv[2], lines)
