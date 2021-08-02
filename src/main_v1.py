import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2

import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import h5py

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import callbacks
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import optimizers
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, TimeDistributed, Flatten, concatenate, Reshape
from keras import metrics

import matplotlib.pyplot as plt
import math
import pickle

img_h=320
img_w=480
img_ch=3

batch_size=1
epochs=10

step=5


max_node=10
max_edge=45


faces_train_path='/home/lfan/Dropbox/JointAttention/node_weight/train/'
faces_train_sf=[join(faces_train_path,sf) for sf in sorted(listdir(faces_train_path)) if isdir(join(faces_train_path,sf))]

faces_validate_path='/home/lfan/Dropbox/JointAttention/node_weight/validate/'
faces_validate_sf=[join(faces_validate_path,sf) for sf in sorted(listdir(faces_validate_path)) if isdir(join(faces_validate_path,sf))]

images_train_path='/home/lfan/Dropbox/runCoAtt/rawData/images/train/'
images_validate_path='/home/lfan/Dropbox/runCoAtt/rawData/images/validate/'


prop_path='/home/lfan/Dropbox/JointAttention/bbx_prop_clear/'

nb_train_samples=0
nb_validate_samples=0
for i in range(len(faces_train_sf)):
    files=[f for f in (listdir(faces_train_sf[i])) if isfile(join(faces_train_sf[i],f))]
    nb_train_samples=nb_train_samples+len(files)

for i in range(len(faces_validate_sf)):
    files=[f for f in (listdir(faces_validate_sf[i])) if isfile(join(faces_validate_sf[i],f))]
    nb_validate_samples=nb_validate_samples+len(files)


model_weights_path='/home/lfan/Dropbox/runCoAtt/experiment/v1/model_weights/'
model_path='/home/lfan/Dropbox/runCoAtt/experiment/v1/model/'


tensorboard_log_dir='/home/lfan/Dropbox/runCoAtt/experiment/v1/tb_log/'

def my_generator_train():
    while True:
        for sfid in range(len(faces_train_sf)):
            faces_train_sf_now = faces_train_sf[sfid]
            vid = faces_train_sf_now.split('/')[-1]

            faces_train_files_now = [f for f in sorted(listdir(faces_train_sf_now)) if isfile(faces_train_sf_now + '/' + f)]

            for i in range((faces_train_files_now) // batch_size + 1):  ## i need to fix the size problem here later...

                x_batch = np.zeros(shape=(batch_size, max_node+max_edge, img_h, img_w, img_ch))
                y_batch = np.zeros(shape=(batch_size, max_node+max_edge))


                if i<(len(faces_train_files_now) // batch_size):
                    SIZE=batch_size
                else:
                    SIZE=len(faces_train_files_now) % batch_size

                for j in range(SIZE):

                    fileid = batch_size * i + j
                    file_now = faces_train_sf_now + '/' + faces_train_files_now[fileid]
                    prop_file_now = prop_path + vid + '/' + str(
                        int(faces_train_files_now[fileid].split('_')[0])) + '.txt'
                    ## prop_file_now may not exist!!!
                    image_now = images_train_path + faces_train_files_now[fileid][:-4] + '.jpg'
                    img = cv2.imread(image_now)

                    with open(file_now, 'r') as file_to_read:
                        face_lines = file_to_read.readlines()
                    if isfile(prop_file_now):
                      with open(prop_file_now, 'r') as prop_to_read:
                        prop_lines = prop_to_read.readlines()

                    face_lines.extend(prop_lines)


                    for k in range(len(face_lines)):
                        if k >= max_node:
                            break
                        line_list = face_lines[k].split()
                        xmin = int(line_list[0])
                        ymin = int(line_list[1])
                        xmax = int(line_list[2])
                        ymax = int(line_list[3])

                        x_batch[j, k, ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
                        if int(line_list[4])!=0:
                            y_batch[j, k] =1


                    edge_cnt=0
                    for m in range(len(face_lines)):
                        for n in range(m+1,len(face_lines)):
                            edge_cnt+=1
                            if edge_cnt>max_edge:
                                break
                            line_list_a = face_lines[m].split()
                            xmin_a = int(line_list_a[0])
                            ymin_a = int(line_list_a[1])
                            xmax_a = int(line_list_a[2])
                            ymax_a = int(line_list_a[3])

                            line_list_b = face_lines[n].split()
                            xmin_b = int(line_list_b[0])
                            ymin_b = int(line_list_b[1])
                            xmax_b = int(line_list_b[2])
                            ymax_b = int(line_list_b[3])

                            x_batch[j, max_node+edge_cnt-1, ymin_a:ymax_a, xmin_a:xmax_a, :] = img[ymin_a:ymax_a, xmin_a:xmax_a, :]
                            x_batch[j, max_node + edge_cnt - 1, ymin_b:ymax_b, xmin_b:xmax_b, :] = img[ymin_b:ymax_b,
                                                                                                   xmin_b:xmax_b, :]

                            if int(line_list_a[4])>0 and (int(line_list_b[5])==int(line_list_a[4])):
                               y_batch[j, max_node + edge_cnt - 1] = 1

                            if int(line_list_b[4]) > 0 and (int(line_list_a[5]) == int(line_list_b[4])):
                               y_batch[j, max_node + edge_cnt - 1] = 1

                yield [x_batch[:,0:max_node,:,:,:],x_batch[:,max_node:(max_node+max_edge),:,:,:]], y_batch


def my_generator_validate():
    while True:
        for sfid in range(len(faces_validate_sf)):
            faces_validate_sf_now = faces_validate_sf[sfid]
            vid = faces_validate_sf_now.split('/')[-1]

            faces_validate_files_now = [f for f in sorted(listdir(faces_validate_sf_now)) if isfile(faces_validate_sf_now + '/' + f)]

            for i in range(len(faces_validate_files_now) // batch_size + 1):  ## i need to fix the size problem here later...

                x_batch = np.zeros(shape=(batch_size, max_node+max_edge, img_h, img_w, img_ch))
                y_batch = np.zeros(shape=(batch_size, max_node+max_edge))


                if i<(len(faces_validate_files_now) // batch_size):
                    SIZE=batch_size
                else:
                    SIZE=len(faces_validate_files_now) % batch_size

                for j in range(SIZE):

                    fileid = batch_size * i + j
                    file_now = faces_validate_sf_now + '/' + faces_validate_files_now[fileid]
                    prop_file_now = prop_path + vid + '/' + str(
                        int(faces_validate_files_now[fileid].split('_')[0])) + '.txt'
                    ## prop_file_now may not exist!!!
                    image_now = images_validate_path + faces_validate_files_now[fileid][:-4] + '.jpg'
                    img = cv2.imread(image_now)

                    with open(file_now, 'r') as file_to_read:
                        face_lines = file_to_read.readlines()

                    if isfile(prop_file_now):
                      with open(prop_file_now, 'r') as prop_to_read:
                        prop_lines = prop_to_read.readlines()

                    face_lines.extend(prop_lines)


                    for k in range(len(face_lines)):
                        if k >= max_node:
                            break
                        line_list = face_lines[k].split()
                        xmin = int(line_list[0])
                        ymin = int(line_list[1])
                        xmax = int(line_list[2])
                        ymax = int(line_list[3])

                        x_batch[j, k, ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
                        if int(line_list[4])!=0:
                            y_batch[j, k] =1


                    edge_cnt=0
                    for m in range(len(face_lines)):
                        for n in range(m+1,len(face_lines)):
                            edge_cnt+=1
                            if edge_cnt>max_edge:
                                break
                            line_list_a = face_lines[m].split()
                            xmin_a = int(line_list_a[0])
                            ymin_a = int(line_list_a[1])
                            xmax_a = int(line_list_a[2])
                            ymax_a = int(line_list_a[3])

                            line_list_b = face_lines[n].split()
                            xmin_b = int(line_list_b[0])
                            ymin_b = int(line_list_b[1])
                            xmax_b = int(line_list_b[2])
                            ymax_b = int(line_list_b[3])

                            x_batch[j, max_node+edge_cnt-1, ymin_a:ymax_a, xmin_a:xmax_a, :] = img[ymin_a:ymax_a, xmin_a:xmax_a, :]
                            x_batch[j, max_node + edge_cnt - 1, ymin_b:ymax_b, xmin_b:xmax_b, :] = img[ymin_b:ymax_b,
                                                                                                   xmin_b:xmax_b, :]

                            if int(line_list_a[4])>0 and (int(line_list_b[5])==int(line_list_a[4])):
                               y_batch[j, max_node + edge_cnt - 1] = 1

                            if int(line_list_b[4]) > 0 and (int(line_list_a[5]) == int(line_list_b[4])):
                               y_batch[j, max_node + edge_cnt - 1] = 1

                yield [x_batch[:, 0:max_node, :, :, :], x_batch[:, max_node:(max_node + max_edge), :, :, :]], y_batch

def train_model():

    node_input = Input(shape=(max_node, img_h, img_w, img_ch), name='node_input')
    edge_input = Input(shape=(max_edge, img_h, img_w, img_ch), name='edge_input')


    x = TimeDistributed(Conv2D(64,(10,10),strides=3,data_format='channels_last',activation=None,use_bias=False))(node_input)
    x = TimeDistributed(Conv2D(32,(10,10),strides=3,data_format='channels_last',activation=None,use_bias=False))(x)
    x = TimeDistributed(Conv2D(10,(3,3), strides=2, data_format='channels_last', activation=None, use_bias=False))(x)
    x = TimeDistributed(Conv2D(5, (3,3), strides=2, data_format='channels_last', activation=None, use_bias=False))(x)
    x = TimeDistributed(Flatten())(x)


    #x = TimeDistributed(Dense(256, activation = 'relu'))(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.4)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.4)(x)
    node_weight = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    y = TimeDistributed(Conv2D(64, (10, 10), strides=3, data_format='channels_last', activation=None, use_bias=False))(edge_input)
    y = TimeDistributed(Conv2D(32, (10, 10), strides=3, data_format='channels_last', activation=None, use_bias=False))(y)
    y = TimeDistributed(Conv2D(10, (3, 3), strides=2, data_format='channels_last', activation=None, use_bias=False))(y)
    y = TimeDistributed(Conv2D(5, (3, 3), strides=2, data_format='channels_last', activation=None, use_bias=False))(y)
    y = TimeDistributed(Flatten())(y)

    #y = TimeDistributed(Dense(256, activation='relu'))(y)
    y = TimeDistributed(Dense(128, activation='relu'))(y)
    y = Dropout(0.4)(y)
    y = TimeDistributed(Dense(64, activation='relu'))(y)
    y = Dropout(0.4)(y)
    edge_weight = TimeDistributed(Dense(1, activation='sigmoid'))(y)

    predicted_weight=concatenate([node_weight,edge_weight],axis=1)

    predicted_weight=Reshape((max_node+max_edge,))(predicted_weight)
    model = Model(inputs=[node_input,edge_input], outputs=predicted_weight)

    model.summary()
    mycallback = []
    tbCallback = TensorBoard(log_dir=tensorboard_log_dir,
                             histogram_freq=0, batch_size=batch_size,
                             write_graph=True, write_images=True)
    mycallback.append(tbCallback)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto')
    model_checkpoint = ModelCheckpoint(
        '/home/lfan/Dropbox/runCoAtt/experiment/v1/checkpoint/model_v1.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=False,
        save_weights_only=True, monitor='val_loss')
    mycallback.append(early_stopping)
    mycallback.append(model_checkpoint)

    sgd = optimizers.SGD(lr=0.1, decay=1e-4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
    history = model.fit_generator(my_generator_train(),
                                     steps_per_epoch=nb_train_samples // batch_size,
                                     epochs=epochs,
                                     validation_data=my_generator_validate(),
                                     validation_steps=nb_validate_samples // batch_size,
                                     callbacks=mycallback)
    model.save_weights(model_weights_path)
    model.save(model_path)


train_model()
