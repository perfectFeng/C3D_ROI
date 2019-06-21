#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

NUM_CLASSES = 12
height = 336
width = 560
channels = 3
num_frames = 32
batch_size = 5
blocks = 8


def f_c3d(_input_data, _dropout, _weights, _biases):

    conv1 = tf.nn.conv3d(_input_data, _weights['wc1'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')
    conv1 = tf.nn.bias_add(conv1, _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')

    # Convolution Layer
    conv2 = tf.nn.conv3d(pool1, _weights['wc2'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2')
    conv2 = tf.nn.bias_add(conv2, _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    # pooling layer
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')

    # Convolution Layer
    conv3 = tf.nn.conv3d(pool2, _weights['wc3a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3a')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = tf.nn.conv3d(conv3, _weights['wc3b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3b')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    # pooling layer
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')

    # Convolution Layer
    conv4 = tf.nn.conv3d(pool3, _weights['wc4a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4a')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = tf.nn.conv3d(conv4, _weights['wc4b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4b')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    # pooling layer
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')

    # Convolution Layer
    conv5 = tf.nn.conv3d(pool4, _weights['wc5a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5a')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = tf.nn.conv3d(conv5, _weights['wc5b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5b')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')

    # pooling layer
    pool5 = tf.nn.max_pool3d(conv5, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool5')

    return pool5


def inference_c3d(f_data, _dropout, f_weights, f_biases, w, b):

    f_data = tf.transpose(f_data, (0, 2, 1, 3, 4, 5))
    f_data = tf.reshape(f_data, (batch_size*8, 32, 112, 112, 3))
    pool5 = f_c3d(f_data, _dropout, f_weights, f_biases)  # 24,1,4,4,512
    pool5 = tf.reshape(pool5, (batch_size*8, 4, 4, 512))

    avg_pool5 = tf.nn.avg_pool(pool5, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')  # 24,1,1,512
    avg_pool5 = tf.reshape(avg_pool5, (batch_size, 8, 512))

    gate = tf.nn.softmax(tf.matmul(tf.reduce_mean(avg_pool5, 2), w) + b)
    avg_pool5 = avg_pool5 * tf.expand_dims(gate, 2)
    avg_pool5 = tf.reshape(avg_pool5, [batch_size, -1])

    dense = tf.nn.relu(tf.matmul(avg_pool5, f_weights['wd2']) + f_biases['bd2'], name='fc2')
    dense = tf.nn.dropout(dense, _dropout)

    out = tf.matmul(dense, f_weights['out']) + f_biases['out']
    return out
