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

import tensorflow as tf
learning_rate = 0.5
x = tf.placeholder(tf.float32, [None, 162])
W = tf.Variable(tf.zeros([162, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 1])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum((y_ - y) * (y_-y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = x_data, y_data
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
Wval = sess.run(W)
Bval = sess.run(b)
print(Wval)
print(Bval)
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
