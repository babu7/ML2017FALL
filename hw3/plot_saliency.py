#!/usr/bin/env python3
import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def deprocess_image(x):
    """
    Hint: Normalize and Clip
    """
     # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    return x


def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=1)
    args = parser.parse_args()
    emotion_classifier = load_model('weights_63509.h5')

    import cnn_train
    face = cnn_train.load_data()
    # face.inputs = np.rint(face.inputs*255).astype(np.int64)
    private_pixels = face.inputs
    # private_pixels,a,b,c = makedata(0)

    input_img = emotion_classifier.input
    for idx in range(100):
        sidx = "%03d" % idx

        plt.figure()
        plt.imshow(private_pixels[idx].reshape(48, 48),cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        # plt.show()
        fig.savefig('img/' + sidx +'_ori.png', dpi=100)

        val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        grads_value = fn([private_pixels[idx].reshape(1,48,48,1), 0])
        heatmap = np.array(grads_value).reshape(48,48)
        see = private_pixels[idx].reshape(48, 48)
        """
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        """
        thres = 0.5
        heatmap = deprocess_image(heatmap)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        # plt.show()
        fig.savefig('img/' + sidx +'_heatmap.png', dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        # plt.show()
        fig.savefig('img/' + sidx +'_see.png', dpi=100)

if __name__ == "__main__":
    main()
