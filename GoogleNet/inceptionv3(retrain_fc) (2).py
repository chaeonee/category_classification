#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:44:56 2018

@author: onee

https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
"""


"""Inception V3 model for Keras.
Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).
# Reference
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
"""
from __future__ import print_function

import warnings
import numpy as np

import os
from PIL import Image
from PIL import ImageFile
import random

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    train_list = []
    train_label = []
    train_zip = []
    
     # Load Training Data
    photo_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/data/photo/'
    painting_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/data/painting/'
    clipart_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/data/clipart/'
    text_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/data/text/'
    figure_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/data/figure/'
    
    photo_file_list = os.listdir(photo_data_dir)
    painting_file_list = os.listdir(painting_data_dir)
    clipart_file_list = os.listdir(clipart_data_dir)
    text_file_list = os.listdir(text_data_dir)
    figure_file_list = os.listdir(figure_data_dir)
    
    for i in range(len(photo_file_list)):
        filenames = photo_data_dir + photo_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(0)
        
    for i in range(len(painting_file_list)):
        filenames = painting_data_dir + painting_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(1)
    
    for i in range(len(clipart_file_list)):
        filenames = clipart_data_dir + clipart_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(2)
        
    for i in range(len(text_file_list)):
        filenames = text_data_dir + text_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(3)
        
    for i in range(len(figure_file_list)):
        filenames = figure_data_dir + figure_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(4)
        
#    kor_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/kor/'
#    eng_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/eng/'
#    chi_data_dir = 'C:/Users/SM-PC/Desktop/img_pic/chi/'
#    
#    kor_file_list = os.listdir(kor_data_dir)
#    eng_file_list = os.listdir(eng_data_dir)
#    chi_file_list = os.listdir(chi_data_dir)
#    
#    for i in range(len(kor_file_list)):
#        filenames = kor_data_dir + kor_file_list[i]
#        train_image = Image.open(filenames)
#        train_image = train_image.convert("RGB")
#        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
#        train_image = np.array(train_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        train_list.append(train_image)
#        
#        train_label.append(0)
#        
#    for i in range(len(eng_file_list)):
#        filenames = eng_data_dir + eng_file_list[i]
#        train_image = Image.open(filenames)
#        train_image = train_image.convert("RGB")
#        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
#        train_image = np.array(train_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        train_list.append(train_image)
#        
#        train_label.append(1)
#        
#    for i in range(len(chi_file_list)):
#        filenames = chi_data_dir + chi_file_list[i]
#        train_image = Image.open(filenames)
#        train_image = train_image.convert("RGB")
#        train_image = train_image.resize((299, 299), Image.ANTIALIAS)
#        train_image = np.array(train_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        train_list.append(train_image)
#        
#        train_label.append(2)
#        
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
    photo_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/data/photo2/'
    painting_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/data/painting2/'
    clipart_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/data/clipart2/'
    text_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/data/text2/'
    figure_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/data/figure2/'
    
    photo_file_list2 = os.listdir(photo_data_dir2)
    painting_file_list2 = os.listdir(painting_data_dir2)
    clipart_file_list2 = os.listdir(clipart_data_dir2)
    text_file_list2 = os.listdir(text_data_dir2)
    figure_file_list2 = os.listdir(figure_data_dir2)
    
    for i in range(len(photo_file_list2)):
        filenames = photo_data_dir2 + photo_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(0)
        
    for i in range(len(painting_file_list2)):
        filenames = painting_data_dir2 + painting_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(1)
        
    for i in range(len(clipart_file_list2)):
        filenames = clipart_data_dir2 + clipart_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(2)
        
    for i in range(len(text_file_list2)):
        filenames = text_data_dir2 + text_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(3)        

    for i in range(len(figure_file_list2)):
        filenames = figure_data_dir2 + figure_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(4)    
        
#    kor_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/kor1/'
#    eng_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/eng1/'
#    chi_data_dir2 = 'C:/Users/SM-PC/Desktop/img_pic/chi1/'
#    
#    kor_file_list2 = os.listdir(kor_data_dir2)
#    eng_file_list2 = os.listdir(eng_data_dir2)
#    chi_file_list2 = os.listdir(chi_data_dir2)
#    
#    for i in range(len(kor_file_list2)):
#        filenames = kor_data_dir2 + kor_file_list2[i]
#        test_image = Image.open(filenames)
#        test_image = test_image.convert("RGB")
#        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
#        test_image = np.array(test_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        test_list.append(test_image)
#        
#        test_label.append(0)   
#        
#    for i in range(len(eng_file_list2)):
#        filenames = eng_data_dir2 + eng_file_list2[i]
#        test_image = Image.open(filenames)
#        test_image = test_image.convert("RGB")
#        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
#        test_image = np.array(test_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        test_list.append(test_image)
#        
#        test_label.append(1)
#        
#    for i in range(len(chi_file_list2)):
#        filenames = chi_data_dir2 + chi_file_list2[i]
#        test_image = Image.open(filenames)
#        test_image = test_image.convert("RGB")
#        test_image = test_image.resize((299, 299), Image.ANTIALIAS)
#        test_image = np.array(test_image, dtype=np.float32)
#        # train_list0.append(train_image) 
#        test_list.append(test_image)
#        
#        test_label.append(2)
        
        
    test_zip = list(zip(test_list, test_label))
    random.shuffle(test_zip)
    
    test_list, test_label = zip(*test_zip)
    test_label = np.asarray(test_label) 
    test_list = np.asarray(test_list)

    model = InceptionV3(include_top=True, weights='imagenet')
    print('Model load')
    
    model.layers.pop()
    
    for layer in model.layers:
        layer.trainable = False
        
    last = model.layers[-1].output
    
    x = Dense(5, activation="softmax")(last)
    
    model = Model(model.input, x)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    
    print("Inception v3")
    
    model.fit(train_list, train_label, epochs=10, batch_size=500)

    score = model.evaluate(test_list, test_label, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    
     # serialize model to JSON
    model_json = model.to_json()
    with open("C:/Users/SM-PC/Desktop/img_pic/inception_50_model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("C:/Users/SM-PC/Desktop/img_pic/inception_50_model.h5")
    print("Saved model to disk")