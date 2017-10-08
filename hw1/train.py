#!/usr/bin/env python3
from pandas import read_csv
import numpy as np
import random

pm25 = 'input_data/train.csv'
class Dataset:
    def __init__(self, x, y):
        self.inputs = x
        self.labels = y

train = test = None
hour = 9
onlypm25 = False
train_all = False
save_model = False
Lambda = 0

# ydata = b + w * xdata + lambda(x**2)
b = -120    # 11
# w = -4  # 0.46
w = None
lr = 1
iteration = 1000

# Store initial values for plotting
b_history = [b]
w_history = [w]

# Custom learning rate
lr_b = 0
lr_w = None

# Iterations
def Grad_Des():
    global b, w, lr, iteration, b_history, w_history, lr_b, lr_w
    import sys
    for i in range(iteration):
        if i % (iteration // 10) == 0:
            print("\b\b\b\b%3d%%" % (i*100/iteration), end='')
            sys.stdout.flush()
        if i == iteration-1:
            print("\b\b\b\b\033[K", end='')
            sys.stdout.flush()

        b_grad = 0.0
        w_grad = np.zeros(train.inputs.shape[1])
        # f(x) = b + w1x1 + w2x2 + Lambda(w1^2 + w2^2)
        # Loss = variance + lambda wi^2
        for n in range(train.labels.shape[0]):
            yn = train.labels[n][0]
            Loss_deri = 2.0 * (yn - b - np.dot(w, train.inputs[n]))
            b_grad = b_grad - Loss_deri
            w_grad = [w_grad[k] - Loss_deri * train.inputs[n][k] + 2*Lambda*w[k] for k in range(len(w))]
            # w_grad = [w_grad_n - Loss_deri * xn for w_grad_n, xn in zip(w_grad, train.inputs[n])]

        lr_b = lr_b + b_grad ** 2
        lr_w = [lr_w_n + w_grad_n ** 2 for lr_w_n, w_grad_n in zip(lr_w, w_grad)]
        # Update parameters.
        b = b - lr/np.sqrt(lr_b) * b_grad
        w = [w[n] - lr/np.sqrt(lr_w[n]) * w_grad[n] for n in range(train.inputs.shape[1])]
        # Store parameters.
        b_history.append(b)
        w_history.append(w)

def calc_error(dataset):
    global w, b
    L = 0
    for Mx, My in zip(dataset.inputs, dataset.labels):
        y_ = My[0]
        L += (y_ - b - np.dot(Mx, w)) ** 2
    L /= dataset.inputs.shape[0]
    return L

def main():
    global train, test, hour, onlypm25, train_all, Lambda, save_model, w, w_grad, lr_w, b

    s = input('select\toption\n* 0\ttrain 70% data\n  1\ttrain all data\nYour choice? ')
    if s and int(s) == 1:
        train_all = True
    s = input('select\tlambda\n* 0\t0\n  1\t1\n  2\t10\n  3\t100\n  4\t1000\n  5\t10000\n  6\t100000\nYour choice? ')
    if s and int(s) >= 1:
        Lambda = 10 ** (int(s)-1)
    s = input('select\thour\n  0\t5hr\n* 1\t9hr\nYour choice? ')
    if s and int(s) == 0:
        hour = 5
    s = input('select\toption\n  0\tonly pm2.5\n* 1\tall features\nYour choice? ')
    if s and int(s) == 0:
        onlypm25 = True
    s = input('Load trained model? (y/n): ')
    if s.lower() == 'y':
        s = input('file name: ')
        try:
            w = np.load(s)['w']
            b = np.load(s)['b']
        except FileNotFoundError:
            print('File Not Found, ignore')

    try:
        df = read_csv(pm25, encoding='big5')
    except UnicodeDecodeError:
        df = read_csv(pm25, encoding='utf8')

    raw = df.values.tolist()
    x_data = []
    for i in range(0, len(raw), 18):
        for j in range(24):
            x_data.append([raw[i][j+3]])
            for k in range(1, 18):
                x_data[i//18*24+j].extend([raw[i + k][j+3]])

    x_data_byday = [[float(j.replace('NR', '0')) for j in i] for i in x_data]
    x_data_byday = np.array(x_data_byday)
    x_data = []
    y_data = []
    for i in range(len(x_data_byday) - hour):
        y_data.append([x_data_byday[i+hour][9]])
        if onlypm25:
            x_data.append(list([x_data_byday[i][9]]))
            for j in range(1, hour):
                x_data[i].extend([x_data_byday[i+j][9]])
        else:
            x_data.append(list(x_data_byday[i]))
            for j in range(1, hour):
                x_data[i].extend(x_data_byday[i+j])

    combined = list(zip(x_data, y_data))
    random.shuffle(combined)
    x_data[:], y_data[:] = zip(*combined)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    if train_all:
        train = Dataset(x_data, y_data)
    else:
        split_idx = len(x_data)*7//10
        train = Dataset(x_data[:split_idx], y_data[:split_idx])
        test = Dataset(x_data[split_idx:], y_data[split_idx:])
    w = np.zeros(train.inputs.shape[1]) if not w else w
    lr_w = np.zeros(train.inputs.shape[1])
    print("train shape: %s" % str(train.inputs.shape))
    print("=== Error before training ===")
    print("train dataset: %.2f" % calc_error(train))
    if test:
        print("test  dataset: %.2f" % calc_error(test))
    Grad_Des()
    print("=== Error after training ===")
    print("train dataset: %.2f" % calc_error(train))
    if test:
        print("test  dataset: %.2f" % calc_error(test))
    s = input('select\toption\n* 0\tdon\'t save model\n  1\tsave model\nYour choice? ')
    if s and int(s) == 1:
        save_model = True
    if save_model:
        s = input('model version: ')
        name = ''
        name += ['features', 'pm25'][onlypm25]
        name += ['-70', '-100'][train_all]
        name += "-%dhr" % hour
        name  = name + '-' + s if s else name
        name += '.npz'
        np.savez(name, b=b, w=w)

if __name__ == '__main__':
    main()
