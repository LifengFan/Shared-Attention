import os
from os import listdir
from os.path import isfile, join
import cv2

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
# dimensions of our images
img_width, img_height =244, 244 #20, 30

top_model_weights_path='faceDirClass.h5'

epochs=10
batch_size=1


facePath='/home/lfan/Dropbox/runCoAtt/darknet/results/test/'
imagePath='/home/lfan/Dropbox/runCoAtt/darknet/mydata/test/images/'
objPath='/home/lfan/Dropbox/runCoAtt/test_results/'
testfiles=[f for f in listdir(facePath) if isfile(join(facePath,f))]
first_flag=1

for i in range(len(testfiles)):
    file_now=facePath+testfiles[i]
    file_obj=objPath+testfiles[i]
    image_now=imagePath+testfiles[i][:-4]+".jpg"
    img_ori=cv2.imread(image_now)
    #img_ori2=load_img(image_now)
    with open(file_now,'r') as file_to_read:
      with open(file_obj,'w') as file_to_write:
        while(True):
            lines = file_to_read.readline()
            if len(lines) == 0:
                break
            line_list=lines.split( )
            cnt=line_list[0]
            label=line_list[1]
            prob=line_list[2]
            xmin=int(line_list[3])
            ymin=int(line_list[4])
            xmax=int(line_list[5])
            ymax=int(line_list[6])

            facebox=img_ori[ymin:(ymax+1),xmin:(xmax+1),:]

            resized_facebox = cv2.resize(facebox, (img_width, img_height))

            resized_facebox = img_to_array(resized_facebox)
            resized_facebox = resized_facebox / 255
            resized_facebox = np.expand_dims(resized_facebox, axis=0)

            num_classes = 20
            # build the VGG16 network
            if first_flag:
               model0 = applications.VGG16(include_top=False, weights='imagenet')

            # get the bottleneck prediction from the pre-trained VGG16 model
            bottleneck_predictions = model0.predict(resized_facebox)

            # build top model
            if first_flag:
               model = Sequential()
               model.add(Flatten(input_shape=bottleneck_predictions.shape[1:]))
               model.add(Dense(256, activation='relu'))
               model.add(Dropout(0.5))
               model.add(Dense(num_classes, activation='sigmoid'))
               model.load_weights(top_model_weights_path)
               first_flag=0

            # use the bottleneck prediction on the top model to get the final classification
            class_predicted = model.predict_classes(bottleneck_predictions)
            inID = class_predicted[0]
            direction = "{}".format(inID + 1)
            #print("Predicted ID: {}, label: {}".format(inID, label))
            file_to_write.writelines("{} {} {} {} {} {} {} {}\n".format(cnt,label,prob,xmin,ymin,xmax,ymax,direction))


















