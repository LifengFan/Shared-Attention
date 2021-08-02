
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
from random import shuffle
from generator import my_generator
from model_generator import my_model
import math
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation,concatenate
import matplotlib.pyplot as plt



# nb_train_samples=3277274
# nb_validate_samples=1718056
# nb_test_samples=1909675
batch_size = 60
epochs = 30

nb_train_pos = 7000
nb_train_neg = 2000
nb_train_samples = 9000

nb_validate_pos = 4500
nb_validate_neg = 1200
nb_validate_samples = 5700

nb_test_pos = 2500
nb_test_neg = 2000
nb_test_samples = 4500

nb_test_pos = 2500
nb_test_neg = 2000
nb_test_samples = 4500

version = 'method8'
model_weights_path = '~/Projects/CVPR2018/NN/experiment/' + version + '/finalweights.hdf5'
model_path = '~/Projects/CVPR2018/NN/experiment/' + version + '/finalmodel.h5'
tensorboard_log_dir = '~/Projects/CVPR2018/NN/experiment/' + version + '/tb_log/'

def lr_decay(epoch):
    initial_lr = 1e-3
    drop = 0.6
    epochs_drop = 2
    lr = initial_lr * math.pow(drop, math.floor(epoch / epochs_drop))

    return lr

def custom_objective(y_true, y_pred):
    lamda = tf.constant(2.0)
    # tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
    # sess = tf.Session()
    # print sess.run(y_pred)
    y_tmp = tf.scalar_mul(lamda,tf.ones_like(y_true))
    y_true=tf.cast(y_true, tf.float32)
    diff = tf.scalar_mul(tf.constant(0.001275510),tf.square(tf.subtract(y_pred , y_true)))
    r = tf.reduce_sum(tf.multiply(y_tmp,diff))
    return r

def train_model():

    mymodel = my_model()
    # mymodel.load_weights('/home/yixin/Projects/CVPR2018/NN/experiment/method7/checkpoint.weights.12-0.13.hdf5', by_name=True)

    mycallback = []
    tbCallback = TensorBoard(log_dir=tensorboard_log_dir,
                             histogram_freq=0, batch_size=batch_size,
                             write_graph=True, write_images=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
    model_checkpoint = ModelCheckpoint(
        '/home/yixin/Projects/CVPR2018/NN/experiment/' + version + '/checkpoint.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=False,
        save_weights_only=True, monitor='val_loss')

    # reducelr=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
    #                                  epsilon=0.0001, cooldown=0, min_lr=1e-6)

    lrscheduler = LearningRateScheduler(lr_decay)

    mycallback.append(tbCallback)
    mycallback.append(early_stopping)
    mycallback.append(model_checkpoint)
    mycallback.append(lrscheduler)
    # mycallback.append(reducelr)

    # sgd = optimizers.SGD(lr=1e-4, decay=1e-3)
    # rmsprop=optimizers.rmsprop(lr=1e-4)
    # adam=optimizers.adam(lr=1e-5)
    mymodel.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'acc'])#custom_objective

    history = mymodel.fit_generator(my_generator(1),
                                    steps_per_epoch=nb_train_samples // batch_size,
                                    epochs=epochs,
                                    validation_data=my_generator(2),
                                    validation_steps=nb_validate_samples // batch_size,
                                    callbacks=mycallback)
    mymodel.save_weights(model_weights_path)
    mymodel.save(model_path)
    score = mymodel.evaluate_generator(my_generator(2), nb_validate_samples // batch_size)
    print("Validation Accuracy= ", score[1])

train_model()
# t = tf.constant(42.0)
# sess = tf.Session()
# print sess.run(t)
# s = np.asarray([2,2,1,1])
# print(np.ones(s))

