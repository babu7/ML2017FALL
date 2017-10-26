#!/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor
from logistic import load_data, _shuffle, normalize, split_valid_set, sigmoid
def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return
def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    # X_train, Y_train = X_all, Y_all

    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106,106))
    sigma2 = np.zeros((106,106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    print('=====Saving Param=====')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param_dict):
        print('Saving %s' % key)
        np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])

    print('=====Validating=====')
    # valid(X_train, Y_train, mu1, mu2, shared_sigma, N1, N2)
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

    return
def infer(X_test, save_dir, output_file):
    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_file)
    # Write output
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     output_path = os.path.join(output_dir, 'prediction.csv')
    with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return
def main(opts):
    # Load feature and label
    X_all, Y_all, X_test = load_data(opts.train_data_path, opts.train_label_path, opts.test_data_path)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)

    # To train or to infer
    if opts.train:
        train(X_all, Y_all, opts.save_dir)
    elif opts.infer:
        infer(X_test, opts.save_dir, opts.output_dir)
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Probabilistic Generative Model for Binary Classification'
             )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='input_data/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='input_data/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='input_data/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_dir', type=str,
                        default='generative_params/', dest='save_dir',
                        help='Path to save the model parameters')
    parser.add_argument('--output_dir', type=str,
                        default='generative_output/', dest='output_dir',
                        help='Path to save output')
    opts = parser.parse_args()
    main(opts)
