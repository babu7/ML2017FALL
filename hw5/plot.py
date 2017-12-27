#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from train import get_emb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import keras.backend.tensorflow_backend as K
from utils import *
K.set_learning_phase(0)

def draw(x, y):
    vis_data = TSNE(n_components=2).fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    plt.colorbar(sc)
    fig = plt.gcf()
    fig.savefig('models/fig.png')

mdict = {'Film-Noir': 0,
        'Sci-Fi': 0,
        'Mystery': 0,
        'Drama': 1,
        'Horror': 2,
        'Romance': 0,
        'Thriller': 2,
        "Children's": 3,
        'Adventure': 3,
        'Action': 0,
        'Comedy': 0,
        'Documentary': 0,
        'Western': 0,
        'War': 0,
        'Musical': 1,
        'Animation': 3,
        'Crime': 2,
        'Fantasy': 0}

# draw(u, m)
def load_movie(data='input_data/movies.csv'):
    lines = open(data).read().rstrip().split('\n')[1:]
    lines = [line.rstrip().split('::') for line in lines]
    lines = [[line[0], line[2].split('|')[0]] for line in lines]
    # mdict = {}
    # for k, v in mdict.items():
    #     print("%11s %s"%(k, v))
    # print(len(mdict))
    # i = 0
    # for line in lines:
    #     cat = line[1]
    #     if not cat in mdict.keys():
    #         mdict[cat] = i
    #         i = i+1
    # print(mdict)
    catid = [mdict[line[1]] for line in lines]
    catid = np.array(catid)
    print(catid.shape)
    lines = np.array(lines)
    return catid
def load_user(data='input_data/user.csv'):
    lines = open(data).read().rstrip().split('\n')

cid = load_movie()
uid, mid, rid = load_train('input_data/train.csv')
d = DataSet(uid=uid, mid=mid, rid=rid, cid=cid)
d.shrink(0.1)

m = load_model('models/d512-dropless-slow/model.h5')
get_emb = K.function([m.layers[1].input], [m.layers[3].output])
res = get_emb([d.mid])[0]
print(res)
