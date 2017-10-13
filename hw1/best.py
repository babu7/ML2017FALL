#!/usr/bin/env python3
from pandas import read_csv
import sys
import os
import numpy as np

argc = len(sys.argv)
# 0: ./best.py   1: test.csv   2: out.csv
# 3: npy array
# 4: npz array (optional)
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
                x_data.append(['1'])
            x_data[i//18].extend([raw[i+k][j+2]])

try:
    model_nr = os.environ['model_nr']
except:
    model_nr = '0'
try:
    model_power = int(os.environ['model_power'])
except:
    model_power = 1

x_data= [[float(j.replace('NR', model_nr)) for j in i] for i in x_data]
if model_power == 2:
    for i in range(len(x_data)):
        x_data[i] = np.array(x_data[i])
        x_data[i] = np.append(x_data[i], x_data[i][1:]**2)
x_data = np.array(x_data)

w = np.load(sys.argv[3])
if argc == 5:
    mean = np.load(sys.argv[4])['mean']
    sd = np.load(sys.argv[4])['sd']
    x_data = (x_data - mean) / sd

y = np.matmul(x_data, w).flatten()

f = open(sys.argv[2], 'w')
print("id,value", file=f)
for i in range(len(y)):
    print("%s,%f" % (raw[i*18][0], y[i]), file=f)
