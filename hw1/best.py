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

try:
    model_nr = os.environ['model_nr']
except:
    model_nr = '0'
try:
    model_power = int(os.environ['model_power'])
except:
    model_power = 1
try:
    onlypm25 = os.environ['pm25']
except:
    onlypm25 = 'f'
if onlypm25.lower() == 't':
    onlypm25 = True
else:
    onlypm25 = False
hour = 9

raw = df.values.tolist()
ids = [raw[i][0] for i in range(0, len(raw), 18)]
raw = [i[2:] for i in raw]
raw = [[float(j.replace('NR', model_nr)) for j in i] for i in raw]
raw = np.array(raw)
# x[number][hour][feature]
x_n_hr_f = []
for i in range(0, len(raw), 18):
    x_n_hr_f.append(raw[i:i+18,:].T)
x_data = []
for i in range(len(x_n_hr_f)):
    x_hr_f = x_n_hr_f[i]
    x_data.append([1])
    if onlypm25:
        toadd = x_hr_f[9-hour:,9]
    else:
        toadd = x_hr_f[9-hour:,:].flatten()
    x_data[i].extend(toadd)
    if model_power == 2:
        x_data[i].extend(toadd**2)
x_data = np.array(x_data)
print("x_data shape %s" % str(x_data.shape))

w = np.load(sys.argv[3])
if argc == 5:
    mean = np.load(sys.argv[4])['mean']
    sd = np.load(sys.argv[4])['sd']
    x_data = (x_data - mean) / sd

y = np.matmul(x_data, w).flatten()

f = open(sys.argv[2], 'w')
print("id,value", file=f)
for i in range(len(y)):
    print("%s,%f" % (ids[i], y[i]), file=f)
