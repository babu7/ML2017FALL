#!/usr/bin/env python3
from pandas import read_csv
import numpy as np

pm25 = 'input_data/train.csv'
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

x_data_byday = [[float(j.replace('NR', '-1')) for j in i] for i in x_data]
x_data = []
y_data = []
for i in range(len(x_data_byday) - 9):
    y_data.append([x_data_byday[i+9][9]])
    x_data.append(x_data_byday[i])
    for j in range(1, 9):
        x_data[i].extend(x_data_byday[i+j])
x_data = np.array(x_data)
y_data = np.array(y_data)

# ydata = b + w * xdata
dimension = 162
b = -120    # 11
# w = -4  # 0.46
w = np.zeros(dimension)
lr = 1
iteration = 100000

# Store initial values for plotting
b_history = [b]
w_history = [w]

# Custom learning rate
lr_b = 0
lr_w = np.zeros(dimension)

# Iterations
for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros(dimension)
    for n in range(len(x_data)):
        yn = y_data[n][0]
        Loss_deri = 2.0 * (yn - b - np.dot(w, x_data[n]))
        b_grad = b_grad - Loss_deri
        w_grad = [w_grad_n - Loss_deri * xn for w_grad_n, xn in zip(w_grad, x_data[n])]

    lr_b = lr_b + b_grad ** 2
    lr_w = [lr_w_n + w_grad_n ** 2 for lr_w_n, w_grad_n in zip(lr_w, w_grad)]
    # Update parameters.
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = [w[n] - lr/np.sqrt(lr_w[n]) * w_grad[n] for n in range(dimension)]
    # Store parameters.
    b_history.append(b)
    w_history.append(w)

print("b: %d\nw: %s" % (b, w))

# # brute force
# # Store all variance
# try_b = np.arange(-200, -100, 1)
# try_w1 = np.arange(-5, 5, 0.1)
# Loss = np.zeros((len(try_b), len(try_w1)))
# X, Y = np.meshgrid(x, y)
# for i in range(len(try_b)):
#     for j in range(len(try_w1)):
#         b = try_b[i]
#         w1 = try_w1[i]
#         Loss[j][i] = 0
#         data_amount = len(x_data) - 9
#
#         # calculate variance
#         for n in range(data_amount):
#             wx_sum = w1 * x_data[29][n] # w * x_data[n]
#             Loss[j][i] = Loss[j][i] + (y_data[n] - b - wx_sum) ** 2
#         Loss[j][i] = Loss[j][i]/len(data_amount)
