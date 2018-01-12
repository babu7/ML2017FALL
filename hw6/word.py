#!/usr/bin/env python3
from collections import Counter
import numpy as np
import jieba
from gensim.models import word2vec
from gensim import models

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontManager
from pylab import mpl
import subprocess
from adjustText import adjust_text

multi_tsne = True
if multi_tsne:
    from MulticoreTSNE import MulticoreTSNE as TSNE
else:
    from sklearn.manifold import TSNE

txt_seg = 'input_data/all_seg.txt'

def segment():
    jieba.set_dictionary('input_data/dict.txt.big')
    with open('input_data/all_sents.txt') as content:
        output = open(txt_seg, 'w')
        for line in content:
            words = jieba.cut(line.rstrip('\n'))
            output.write(" ".join(words) + '\n')

def train():
    sentences = word2vec.Text8Corpus(txt_seg)
    model = word2vec.Word2Vec(sentences, size=250)
    # Save our model.
    model.save("input_data/med250.model.bin")

def plot():
    mpl.rcParams['font.family'] = ['Noto Sans CJK TC']  # set default font
    mpl.rcParams['axes.unicode_minus'] = False          # use the ascii hyphen for '-'

    m = models.Word2Vec.load('input_data/med250.model.bin')
    cnt = Counter(open(txt_seg).read().split())
    matches = [x for x in cnt.most_common() if x[1]>=3000 and x[1]<=6000]
    matches = [x[0] for x in matches if x[0] != '「' and x[0] != '」']
    embs = m.wv[matches]
    print('TSNE ...')
    if multi_tsne:
        embs2d = TSNE(n_components=2, n_iter=4000, n_jobs=8).fit_transform(embs)
    else:
        embs2d = TSNE(n_components=2).fit_transform(embs)
    fig, ax = plt.subplots()
    ax.plot(embs2d[:,0], embs2d[:,1], 'bo')
    texts = []
    for x, y, s in zip(embs2d[:,0], embs2d[:,1], matches):
        texts.append(ax.text(x, y, s))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    fig.tight_layout()
    fig.savefig('word.png')
plot()
