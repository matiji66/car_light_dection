#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Spring

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
# import MNIST_data.mnist_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 100  # 隐藏层数

BATCH_SIZE = 120  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99


# def get_weight_variable(shape, regularizer):
#     weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
#     if regularizer != None:
#         tf.add_to_collection('losses', regularizer(weights))
#     return weights
#
#
# def inference(input_tensor, regularizer):
#     with tf.variable_scope('layer1'):
#
#         weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
#         biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
#
#     with tf.variable_scope('layer2'):
#         weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
#         biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
#         layer2 = tf.matmul(layer1, weights) + biases
#
#     return layer2
#
def get_weight_variable(shape, regularizer):

    weights = tf.get_variable("weights", shape, initializer=tf.truncate_normal_initializer(stddev=0.1))

    if regularizer != None:
        weights = tf.add_to_collection('losses',regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(name='biases', initializer=tf.constant_initializer(0.0),shape=[LAYER1_NODE])
        layer1 = tf.matmul(input_tensor,weights)+ biases
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(name='biases', initializer=tf.constant_initializer(0.0),shape=[OUTPUT_NODE])
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


# def train(mnist):
#     x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
#     y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
#     regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#     y = mnist_inference.inference(x, regularizer)
#     global_step = tf.Variable(0, trainable=False)
#
#     variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#     learning_rate = tf.train.exponential_decay(
#         LEARNING_RATE_BASE,
#         global_step,
#         mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
#         staircase=True)
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#     with tf.control_dependencies([train_step, variables_averages_op]):
#         train_op = tf.no_op(name='train')
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#
#         for i in range(TRAINING_STEPS):
#             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
#             if i % 1000 == 0:
#                 print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#                 saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE], name="x")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE], name='y_')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x,regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, regularizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
