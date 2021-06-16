# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:58:51 2018

@author: onee
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from collections import namedtuple
# from math import sqrt

import numpy as np
import tensorflow as tf

import os
from PIL import Image
import random
import csv

X_FEATURE = 'x'  # Name of the input feature.

def resnet_18_model(features, labels, mode):
    
    filter_set = [128, 256, 512]
    
    #x=tf.placeholder(tf.float32, [None,224,224,3])
    x = features[X_FEATURE]
    input_shape = x.get_shape().as_list()
    
#    # Reshape the input into the right shape if it's 2D tensor
#    if len(input_shape) == 2:
#        ndim = int(sqrt(input_shape[1]))
#        x = tf.reshape(x, [-1, ndim, ndim, 1])
    
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
        
    with tf.variable_scope('conv_layer2'):
        net = tf.layers.conv2d(
                net,
                filters=64,
                kernel_size=1,
                padding='valid',
                activation=tf.nn.relu)
         
    for i in range(2):
        with tf.variable_scope('conv_layer2_'+str(i+1)):
            conv = tf.layers.conv2d(
                    net,
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)

        with tf.variable_scope('conv_layer2__'+str(i+1)):
            conv = tf.layers.conv2d(
                    conv,
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)            
            
        net = net + conv
         
    for i in range(3):
        with tf.variable_scope('conv_layer'+str(i+3)):
            net = tf.layers.conv2d(
                    net,
                    filters=filter_set[i],
                    kernel_size=1,
                    padding='valid',
                    strides=(2, 2),
                    activation=tf.nn.relu)
             
        for j in range(2):
            with tf.variable_scope('conv_layer'+str(i+3)+'_'+str(j+1)):
                conv = tf.layers.conv2d(
                        net,
                        filters=filter_set[i],
                        kernel_size=3,
                        padding='same',
                        activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv)
            
            with tf.variable_scope('conv_layer'+str(i+3)+'__'+str(j+1)):
                conv = tf.layers.conv2d(
                        conv,
                        filters=filter_set[i],
                        kernel_size=3,
                        padding='same',
                        activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv)            
                
            net = net + conv
            
            
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
    
    #Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
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
    
    
    
    #Load Test Data
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
    
    classifier = tf.contrib.learn.Estimator(model_fn=resnet_18_model)
    
    train_input_fn = tf.contrib.learn.io.numpy_input_fn(
            x={X_FEATURE: train_list},
            y=train_label.astype(np.int32),
            batch_size=20,
            num_epochs=10, #여기
            shuffle=True)
    
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