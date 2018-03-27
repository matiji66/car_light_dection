#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @author:Spring

import tensorflow as tf
from numpy.random import RandomState


batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([batch_size,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2

cross_entropy = -tf.reduce_mean( y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[x1**3 + x2**2 ] for (x1, x2) in X]
Y

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 50000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        xx = X[start:end]
        yy = Y[start:end]

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            start_index = rdm.random_integers(low=0, high=120)
            print('start_index ==',start_index)
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X[start_index:start_index +batch_size], y_: Y[start_index:start_index +batch_size]})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

            # 输出训练后的参数取值。
            print("\n")
            print("w1:", sess.run(w1))
            print("w2:", sess.run(w2))
