#!/usr/bin/env python3
from pandas import read_csv
import numpy as np
from numpy.linalg import inv
import os

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
power = 1
s_sd = None
s_mean = None

# ydata = b + w * xdata + lambda(x**2)
w = None
lr = 1
iteration = 1000

# Custom learning rate
lr_w = None

def closed_form_sol():
    global w
    x = train.inputs
    y = train.labels
    w = np.matmul(np.matmul(inv(np.matmul(x.T, x)), x.T), y)

# Iterations
def Grad_Des():
    global w, lr, iteration, lr_w
    l_rate = 10
    repeat = 100000
    x = train.inputs
    y = train.labels
    s_gra = np.zeros((x.shape[1], 1))

    for i in range(repeat):
        hypo = np.matmul(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / x.shape[0]
        cost_a  = np.sqrt(cost)
        gra = np.matmul(x.T,loss) - Lambda * w
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra/ada
        # print ('\r\033[Kiteration: %d | Cost: %f' % ( i,cost_a), end='')
    print('')
#     for i in range(iteration):
#         if i % (iteration // 10) == 0:
#             print("\b\b\b\b%3d%%" % (i*100/iteration), end='')
#             sys.stdout.flush()
#         if i == iteration-1:
#             print("\b\b\b\b\033[K", end='')
#             sys.stdout.flush()
#
#         w_grad = np.zeros(train.inputs.shape[1])
#         # f(x) = b + w1x1 + w2x2 + Lambda(w1^2 + w2^2)
#         # Loss = variance + lambda wi^2
#         for n in range(train.labels.shape[0]):
#             yn = train.labels[n][0]
#             Loss_deri = 2.0 * (yn - np.matmul(train.inputs, w).sum())
#             # b_grad = b_grad - Loss_deri
#             w_grad = [w_grad[k] - Loss_deri * train.inputs[n][k] + 2*Lambda*w[k] for k in range(len(w))]
#             # w_grad = [w_grad_n - Loss_deri * xn for w_grad_n, xn in zip(w_grad, train.inputs[n])]
#
#         # lr_b = lr_b + b_grad ** 2
#         lr_w = [lr_w_n + w_grad_n ** 2 for lr_w_n, w_grad_n in zip(lr_w, w_grad)]
#         # Update parameters.
#         # b = b - lr/np.sqrt(lr_b) * b_grad
#         w = [w[n] - lr/np.sqrt(lr_w[n]) * w_grad[n] for n in range(train.inputs.shape[1])]

def calc_error(dataset):
    x = dataset.inputs
    y = dataset.labels
    hypo = np.matmul(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / x.shape[0]
    cost_a  = np.sqrt(cost)
    return cost_a

def feature_scaling(x):
    global s_sd , s_mean
    s_mean = np.mean(x, axis=0)
    s_sd = np.std(x, axis=0)
    # x[0] is x ** 0
    s_mean[0] = 0
    s_sd[0] = 1
    x = (x - s_mean) / s_sd
    return x

def main():
    global train, test, hour, onlypm25, train_all, Lambda, save_model, w, w_grad, lr_w, power
    scaling = False

    s = input('select\toption\n* 0\ttrain 70% data\n  1\ttrain all data\nYour choice? ')
    if s and int(s) == 1:
        train_all = True
    s = input('power of x? (*1 or 2): ')
    if s and int(s) == 2:
        power = 2
    s = input('select\tlambda\n* 0\t0\n  1\t0.1\n  2\t0.01\n  3\t0.001\n  4\t0.0001\nYour choice? ')
    if s and int(s) >= 1:
        Lambda = 10 ** (-1*int(s))
    s = input('select\thour\n  0\t5hr\n* 1\t9hr\nYour choice? ')
    if s and int(s) == 0:
        hour = 5
    s = input('select\toption\n  0\tonly pm2.5\n* 1\tall features\nYour choice? ')
    if s and int(s) == 0:
        onlypm25 = True
    s = input('scaling? (y/*n): ')
    if s.lower() == 'y':
        scaling = True
    s = input('Load trained model? (y/*n): ')
    if s.lower() == 'y':
        s = input('file name: ')
        try:
            w = np.load(s)
        except FileNotFoundError:
            print('File Not Found, ignore')

    try:
        df = read_csv(pm25, encoding='big5')
    except UnicodeDecodeError:
        df = read_csv(pm25, encoding='utf8')

    try:
        model_nr = os.environ['model_nr']
    except:
        model_nr = '0'

    raw = df.values.tolist()
    raw = [i[3:] for i in raw]
    raw = [[float(j.replace('NR', model_nr)) for j in i] for i in raw]
    raw = np.array(raw)
    # x[month][hr][feature]
    x_m_hr_f = []
    for i in range(12):
        x_m_hr_f.append(raw[i*18*20:i*18*20+18,:].T)
        for j in range(1, 20):
            toadd = raw[i*18*20+j*18:i*18*20+(j+1)*18,:].T
            x_m_hr_f[-1] = np.concatenate((x_m_hr_f[-1], toadd))

    x_data = []
    y_data = []
    for month in (range(12)):
        x_hr_f = x_m_hr_f[month]
        for i in range(x_hr_f.shape[0] - hour):
            y_data.append([x_hr_f[i+hour][9]])
            # x_data
            # [[1, hr0 temp, hr0 ch4, ... , hr1 ... , hr0 temp**2, hr0 ch4**2, ...], [next data...]]
            #   ^ bias                                ^^^ this may not exist with lower power
            x_data.append([1])
            if onlypm25:
                toadd = x_hr_f[i:i+hour,9].flatten()
            else:
                toadd = x_hr_f[i:i+hour,:].flatten()
            x_data[-1] = np.append(x_data[-1], toadd)
            if power == 2:
                x_data[-1] = np.append(x_data[-1], toadd**2)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    if scaling:
        x_data = feature_scaling(x_data)
    permutation = np.random.permutation(x_data.shape[0])
    x_data = x_data[permutation]
    y_data = y_data[permutation]
    if train_all:
        train = Dataset(x_data, y_data)
    else:
        split_idx = x_data.shape[0]*7//10
        train = Dataset(x_data[:split_idx], y_data[:split_idx])
        test = Dataset(x_data[split_idx:], y_data[split_idx:])
    w = np.zeros((train.inputs.shape[1], 1)) if w is None else w
    lr_w = np.zeros((train.inputs.shape[1], 1))
    print("train shape: %s" % str(train.inputs.shape))
    print("=== Error before training ===")
    print("train dataset: %f" % calc_error(train))
    if test:
        print("test  dataset: %f" % calc_error(test))
    Grad_Des()
    # closed_form_sol()
    print("=== Error after training ===")
    print("train dataset: %f" % calc_error(train))
    if test:
        print("test  dataset: %f" % calc_error(test))
    s = input('select\toption\n* 0\tdon\'t save model\n  1\tsave model\nYour choice? ')
    if s and int(s) == 1:
        save_model = True
    if save_model:
        s = input('model version: ')
        name = ''
        name += ['features', 'pm25'][onlypm25]
        name += ['-70', '-100'][train_all]
        name += "-%dhr" % hour
        name += "-x%d" % power
        name += '-nr' + model_nr
        name  = name + '-' + s if s else name
        name += '.npy'
        np.save(name, w)
        if scaling:
            name = name.replace('.npy', '-scaling.npz')
            np.savez(name, mean=s_mean, sd=s_sd)

if __name__ == '__main__':
    main()
