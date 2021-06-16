# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:59:38 2018

@author: onee
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from collections import namedtuple
from math import sqrt

import numpy as np
import tensorflow as tf

import os
from PIL import Image
import random
import csv

X_FEATURE = 'x'  # Name of the input feature.

def res_net_model(features, labels, mode):
    """Builds a residual network."""
    
    iterators = [4, 6, 3]
    filter_set = [[128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    
    #x=tf.placeholder(tf.float32, [None,224,224,3])
    x = features[X_FEATURE]
    input_shape = x.get_shape().as_list()
    
    # Reshape the input into the right shape if it's 2D tensor
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])
    
    #First Convolution Layer
    with tf.variable_scope('conv_layer1'):
        net = tf.layers.conv2d(
                x, 
                filters=64,
                kernel_size=7,
                strides=(2, 2),
                padding='same',
                activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net)
        
        # Max pool
        net = tf.layers.max_pooling2d(
                net,
                pool_size=3,
                strides=2,
                padding='same')
        
    # print(net.get_shape().as_list())
    
    # First chain of resnets
    with tf.variable_scope('conv_layer2'):
        net = tf.layers.conv2d(
                net,
                filters=256,
                kernel_size=1,
                padding='valid')
        
    for i in range(0,3):
        # 1x1 convolution responsible for reducing dimension
        with tf.variable_scope('3-'+str(i)+'/conv_in'):
            conv = tf.layers.conv2d(
                    net,
                    filters=64,
                    kernel_size=1,
                    padding='valid',
                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            
        with tf.variable_scope('3-'+str(i)+'/conv_bottleneck'):
            conv = tf.layers.conv2d(
                    conv,
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
        
        # 1x1 convolution responsible for restoring dimension
        with tf.variable_scope('3-'+str(i)+'/conv_out'):
            conv = tf.layers.conv2d(
                    conv,
                    filters=256,
                    kernel_size=1,
                    padding='valid',
                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            
        # shortcut connections that turn the network into its counterpart
        # residual function (identity shortcut)
        net = conv + net
        
    for i in range(0,3):
        with tf.variable_scope('conv_layer'+str(3+i)):
            net = tf.layers.conv2d(
                    net,
                    filters=filter_set[i][2],
                    kernel_size=1,
                    strides=(2, 2),
                    padding='valid')
        
        for j in range(0, iterators[i]):
            # 1x1 convolution responsible for reducing dimension
            with tf.variable_scope(str(4+i)+'-'+str(j)+'/conv_in'):
                conv = tf.layers.conv2d(
                        net,
                        filters=filter_set[i][0],
                        kernel_size=1,
                        padding='valid',
                        activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv)
                
            with tf.variable_scope(str(4+i)+'-'+str(j)+'/conv_bottleneck'):
                conv = tf.layers.conv2d(
                        conv,
                        filters=filter_set[i][1],
                        kernel_size=3,
                        padding='same',
                        activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv)
                
            # 1x1 convolution responsible for restoring dimension
            with tf.variable_scope(str(4+i)+'-'+str(j)+'/conv_out'):
                conv = tf.layers.conv2d(
                        conv,
                        filters=filter_set[i][2],
                        kernel_size=1,
                        padding='valid',
                        activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv)
                
            # shortcut connections that turn the network into its counterpart
            # residual function (identity shortcut)
            net = conv + net
            
    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(
            net,
            ksize=[1, net_shape[1], net_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
    
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    
    logits = tf.layers.dense(net, 2, activation=None)
    
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
                }
        return tf.contrib.learn.ModelFnOps(mode, predictions=predictions)
    
    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:# or mode == tf.estimator.ModeKeys.EVAL:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.contrib.learn.ModelFnOps(mode, loss=loss, train_op=train_op)
    
    # Compute evaluation metrics.
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=predicted_classes)
            }
    
 
    predictions = {
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
                }
 
    return tf.contrib.learn.ModelFnOps(
            mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)


def main(unused_args):
    train_list = []
    train_label = []
    train_zip = []
    
    # Load Training Data
    photo_data_dir = './data/photo/'
    painting_data_dir = './data/painting/'
    
    photo_file_list = os.listdir(photo_data_dir)
    painting_file_list = os.listdir(painting_data_dir)        

    for i in range(len(photo_file_list)):
        filenames = photo_data_dir + photo_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((224, 224), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(0)
        
    for i in range(len(painting_file_list)):
        filenames = painting_data_dir + painting_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((224, 224), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(1)
        
    train_zip = list(zip(train_list, train_label))
    random.shuffle(train_zip)
    
    train_list, train_label = zip(*train_zip)
    train_label = np.asarray(train_label) 
    train_list = np.asarray(train_list)
    
    
    
    ###############Test####################
    test_list = []
    test_label = []
    test_zip = []
    
    #Load Test Data
    photo_data_dir2 = './data/photo_test/'
    painting_data_dir2 = './data/painting_test/'
    
    photo_file_list2 = os.listdir(photo_data_dir2)
    painting_file_list2 = os.listdir(painting_data_dir2)        
        
    for i in range(len(photo_file_list2)):
        filenames = photo_data_dir2 + photo_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((224, 224), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(0)
        
    for i in range(len(painting_file_list2)):
        filenames = painting_data_dir2 + painting_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((224, 224), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(1)
        
    
    test_zip = list(zip(test_list, test_label))
    random.shuffle(test_zip)
    
    test_list, test_label = zip(*test_zip)
    test_label = np.asarray(test_label) 
    test_list = np.asarray(test_list)
    
    # Create a new resnet classifier.
    classifier = tf.contrib.learn.Estimator(model_fn=res_net_model)
    
    #tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.
    
    # Train model and save summaries into logdir.
    train_input_fn = tf.contrib.learn.io.numpy_input_fn(
            x={X_FEATURE: train_list},
            y=train_label.astype(np.int32),
            batch_size=20,
            num_epochs=10, #여기
            shuffle=True)
    
    # Calculate accuracy.
    test_input_fn = tf.contrib.learn.io.numpy_input_fn(
            x={X_FEATURE: test_list},
            y=test_label.astype(np.int32),
            batch_size=5,
            num_epochs=3,
            shuffle=False)
    
    classifier.fit(input_fn=train_input_fn, steps=100)
#    train_scores = classifier.evaluate(input_fn=train_input_fn)
#    print('###############################################')
#    print('Train_loss: {0:f}'.format(train_scores['loss'])+'  Train_accuracy: {0:f}'.format(train_scores['accuracy']))    
#    print('###############################################')
       
    train_scores = classifier.evaluate(input_fn=train_input_fn)
    eval_scores = classifier.evaluate(input_fn=test_input_fn, steps=12) #step 함수가 최대 몇번 실행되는지
    
    print('###############################################')
    print('Train_loss: {0:f}'.format(train_scores['loss'])+'  Train_accuracy: {0:f}'.format(train_scores['accuracy']))    
    print('###############################################')
    
    print('###############################################')
    print('Test_loss: {0:f}'.format(eval_scores['loss'])+'  Test_accuracy: {0:f}'.format(eval_scores['accuracy']))
    print('###############################################')
    
if __name__ == '__main__':
    tf.app.run()