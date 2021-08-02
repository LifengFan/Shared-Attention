import os
from os import listdir
from os.path import isfile, join, isdir
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import h5py
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import callbacks
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.utils.np_utils import to_categorical
import cv2
import matplotlib.pyplot as plt
import math
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras import optimizers
import keras.losses
import tensorflow as tf
from keras import backend as K



img_width, img_height = 224, 224
epochs = 10
batch_size = 36

weights=tf.convert_to_tensor(np.load('weight.npy'))

# def myloss(y_true, y_pred):
#     return 0.5 - np.abs(np.abs(y_true - y_pred) - 0.5)


def myloss(y_true,y_pred):
    #print(np.asarray(y_true)/0.05)
    class_id=tf.ceil(np.asarray(y_true+K.epsilon())/0.05)
    class_id=K.cast(class_id,'int32')
    #boolean_id0=K.cast(K.equal(class_id,0),K.floatx())
    #print(boolean_id0)
    weight=K.cast(K.gather(weights,class_id),'float32')
    # print(weight.dtype)
    # print(y_true.dtype)
    # print(y_pred.dtype)
    loss=(0.5-np.abs(np.abs(y_true-y_pred)-0.5))*(1-weight)*(1-weight)
    #loss = (0.5 - np.abs(np.abs(y_true - y_pred) - 0.5))
    return loss

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Create your own input format (here 3x200x200)
input = Input(shape=(224, 224, 3), name='image_input')

# Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

# Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)

# Create your own model
my_model = Model(input=input, output=x)

# In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# my_model.summary()
mycallback = []
tbCallback = TensorBoard(log_dir='/home/lfan/Dropbox/runCoAtt/vgg16/regress/graph_justface_newloss2/',
                         histogram_freq=0, batch_size=batch_size,
                         write_graph=True, write_images=True)
mycallback.append(tbCallback)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto')
model_checkpoint = ModelCheckpoint(
    '/home/lfan/Dropbox/runCoAtt/vgg16/regress/model_justface_newloss2/gazefollow_regress.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    save_best_only=False,
    save_weights_only=True, monitor='val_loss')
mycallback.append(early_stopping)
mycallback.append(model_checkpoint)
sgd = optimizers.SGD(lr=0.01, decay=1e-4)
my_model.compile(optimizer=sgd, loss=myloss, metrics=['mae', myloss])
my_model.load_weights('/home/lfan/Dropbox/runCoAtt/vgg16/regress/model_justface_newloss2/gazefollow_regress_finalweights.h5')

def compute_score(testid):
    datapath = '/home/lfan/Dropbox/runCoAtt/vgg16/mydata/train/' + str(testid) + '/'
    testpics = [join(datapath, i) for i in listdir(datapath) if isfile(join(datapath, i))]

    score_all = []
    for i in range(len(testpics) // batch_size):
        x_batch = np.zeros(shape=(batch_size, 224, 224, 3))
        # y_batch = np.zeros(shape=(batch_size, 1))
        for j in range(batch_size):
            line = testpics[i * batch_size + j]
            img = cv2.imread(line)
            imgpatch = cv2.resize(img, (img_width, img_height))
            imgpatch = np.float32(imgpatch) / 255
            # direction = 11.0 / 20
            # direction = direction / (math.pi * 2)
            x_batch[j] = imgpatch
            # y_batch[j] = direction

        score = my_model.predict(x_batch, batch_size=batch_size)
        # imagepath_now=testpics[i]
        # print(score)
        score_all.append(score)

    print(' ')
    print('testid: '+str(testid))
    #print('score_all: ')
    #print score_all
    print('average score: ')
    score_all = np.asarray(score_all)
    print(np.mean(score_all))
    print('std: ')
    print(np.std(score_all))

for testid in range(1,21):
    compute_score(testid)
