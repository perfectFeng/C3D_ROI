#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import random
import sys
import os
import event_read
import c3d_model
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"

model_filename = "./chckPts/save30645.ckpt"# sports1m_finetuning_ucf101.model"
start_step = 0

batch_size = c3d_model.batch_size
num_frames = c3d_model.num_frames
height = c3d_model.height
width = c3d_model.width
channels = c3d_model.channels
n_classes = c3d_model.NUM_CLASSES

max_iters = 8


def _variable_with_weight_decay(name, shape, wd):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def calc_reward(logit):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit)
    )
    tf.summary.scalar(
        'cross_entropy',
        cross_entropy_mean
    )
    weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))

    tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def evaluate():
    nextX, nextY = event_read.readTestFile(batch_size, num_frames)
    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
    r = sess.run(accuracy, feed_dict=feed_dict)

    print("ACCURACY: " + str(r))


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        labels_placeholder = tf.placeholder(tf.int64, shape=batch_size)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 32, 8, 112, 112, channels))

        with tf.variable_scope('var_name') as var_scope:
            weights = {
                  'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                  'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                  'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                  'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                  'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                  'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                  'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                  'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                  'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                  'out': _variable_with_weight_decay('wout', [4096, n_classes], 0.0005)

                  }
          
            biases = {
                  'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                  'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                  'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                  'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                  'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                  'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                  'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                  'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                  'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                  'out': _variable_with_weight_decay('bout', [n_classes], 0.000),
                  }
        with tf.variable_scope('gate'):
            w = _variable_with_weight_decay('w', [8, 8], 0.0005)
            b = _variable_with_weight_decay('b', [8], 0.000)

        outputs = c3d_model.inference_c3d(
            inputs_placeholder, 
            0.5,
            weights,
            biases,
            w,
            b)

        loss = calc_reward(outputs)

        param = tf.trainable_variables()
        var1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gate')
        var2 = [i for i in param if i not in var1]

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        train_op1 = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(loss, var_list=var2)
        train_op2 = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(loss, var_list=var1)
        train_op = tf.group(train_op1, train_op2)
        null_op = tf.no_op()

        accuracy = tower_acc(outputs, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            # init = tf.global_variables_initializer()
            # sess.run(init)
            '''
            restore = tf.train.Saver({
                                'var_name/wc1':weights['wc1'],'var_name/bc1':biases['bc1'],
                                'var_name/wc2':weights['wc2'],'var_name/bc2':biases['bc2'],
                                'var_name/wc3a':weights['wc3a'],'var_name/bc3a':biases['bc3a'],
                                'var_name/wc3b':weights['wc3b'],'var_name/bc3b':biases['bc3b'],
                                'var_name/wc4a':weights['wc4a'],'var_name/bc4a':biases['bc4a'],
                                'var_name/wc4b':weights['wc4b'],'var_name/bc4b':biases['bc4b'],
                                'var_name/wc5a':weights['wc5a'],'var_name/bc5a':biases['bc5a'],
                                'var_name/wc5b':weights['wc5b'],'var_name/bc5b':biases['bc5b'],
                                'var_name/wd2':weights['wd2'],'var_name/bd2':biases['bd2'],
                                'var_name/wout': weights['out'], 'var_name/bout': biases['out'],
                                'gate/w': w, 'gate/b': b
                                })'''
            saver.restore(sess, model_filename)

            summary_writer = tf.summary.FileWriter(summaryFolderName, graph=sess.graph)
            # training
            for epoch in range(max_iters):

                lines = event_read.readFile()

                for batch in range(int(len(lines) / batch_size)):

                    start_time = time.time()
                    nextX, nextY = event_read.readTrainData(batch, lines, batch_size, num_frames)

                    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}

                    _, summary, l, acc = sess.run([train_op, merged, loss, accuracy], feed_dict=feed_dict)

                    duration = time.time() - start_time

                    print('epoch-step %d-%d: %.3f sec' % (epoch, batch, duration))

                    if batch % 10 == 0:
                        saver.save(sess,
                                   save_dir + save_prefix + str(epoch * int(len(lines) / batch_size) + batch) + ".ckpt")
                        print('loss:', l, '---', 'acc:', acc)
                        summary_writer.add_summary(summary, epoch * int(len(lines) / batch_size) + batch)
                        evaluate()
