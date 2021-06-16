# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:31:04 2018

@author: onee
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from PIL import Image
import numpy as np

import yaml


def categResult(filenames):
    test_list = []
    
    test_image = Image.open(filenames)
    test_image = test_image.convert("RGB")
    test_image = test_image.resize((299, 299), Image.ANTIALIAS)
    test_image = np.array(test_image, dtype=np.float32)
    test_list.append(test_image)

    test_list = np.asarray(test_list)
    

    from keras.models import model_from_json
    
    
    model_path = yaml.load(open('model_path.yml'))
    
    # load json and create model
    json_file = open(model_path['json_path'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(model_path['h5_path'])#ht_model
    print("Loaded model from disk")
    
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    
    y_prob = loaded_model.predict(test_list)
    y_classes = y_prob.argmax(axis=-1)
    y_classes = y_classes.tolist()
    y_prob = y_prob.tolist()


    for i in range(len(y_prob)):        
        if y_classes[i] == 0:
            return 0 # 사진
            
        elif y_classes[i] == 1:
            return 1 # 그림
            
        elif y_classes[i] == 2:
            return 2 # 도형
            
        elif y_classes[i] == 3:
            return 3 # 어문
            
        elif y_classes[i] == 4:
            return 4 # 삽화
