#!/usr/bin/env python3
import sys
from keras.models import Sequential, load_model
from sklearn.cluster import KMeans
from utils import *

m = load_model('%s/model.h5' % sys.argv[1])
enc = []
n = (len(m.layers)-1) // 2
for i in range(n):
    enc.append(Sequential(m.layers[0:i+2]))
print('n_encoder: %d' % n)

img = load_img()
img_encoded = [e.predict(img) for e in enc]
kmeans = [KMeans(n_clusters=2, n_init=20, max_iter=500, n_jobs=8).fit(i_e) for i_e in img_encoded]
y = [k.labels_ for k in kmeans]
y = [get_test(yi, load_test_ls()) for yi in y]
for i in range(n):
    with open('%s/cluster_%d.txt'%(sys.argv[1], i), 'w') as f:
        print('ID,dataset,dataset', file=f)
        for j in range(y[i].shape[0] // 2):
            print("%d,%d,%d" % (j, y[i][j*2], y[i][j*2+1]), file=f)
    with open('%s/predict_%d.csv'%(sys.argv[1], i), 'w') as f:
        print("ID,Ans", file=f)
        for j in range(y[i].shape[0] // 2):
            if y[i][j*2] == y[i][j*2+1]:
                print("%d,1"%j, file=f)
            else:
                print("%d,0"%j, file=f)
