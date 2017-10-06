#!/usr/bin/env python3
from pandas import read_csv
import sys
import numpy as np

try:
    df = read_csv(sys.argv[1], encoding='big5', header=None)
except UnicodeDecodeError:
    df = read_csv(sys.argv[1], encoding='utf8', header=None)

raw = df.values.tolist()
x_data = []

for i in range(0, len(raw), 18):
    for j in range(9):
        for k in range(18):
            if j == 0 and k == 0:
                x_data.append([raw[i+k][j+2]])
            else:
                x_data[i//18].extend([raw[i+k][j+2]])
x_data= [[float(j.replace('NR', '0')) for j in i] for i in x_data]
x_data = np.array(x_data)

npz = np.load('features-100-9hr.npz')
b = npz['b']
w = npz['w']
y = [np.dot(w, x) + b for x in x_data]

f = open(sys.argv[2], 'w')
print("id,value", file=f)
for i in range(len(y)):
    print("%s,%d" % (raw[i*18][0], y[i]), file=f)
