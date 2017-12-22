#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from train import get_emb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

m = load_model('models/dim512/model.h5')
u, m = get_emb(m)
def draw(x, y):
    vis_data = TSNE(n_components=2).fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    plt.colorbar(sc)
    fig = plt.gcf()
    fig.savefig('models/fig.png')

draw(u, m)
