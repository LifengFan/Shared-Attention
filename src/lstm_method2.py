from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Activation, Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, Merge
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import LSTM, Bidirectional, TimeDistributed, Merge, ConvLSTM2D
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from generator import my_generator
from keras.layers.normalization import BatchNormalization
import math

version = 'lstm_2'

model_weights_path = '/home/lfan/Dropbox/runCoAtt/new_experiment/' + version + '/finalweights.hdf5'
model_path = '/home/lfan/Dropbox/runCoAtt/new_experiment/' + version + '/finalmodel.h5'

time_step = 10
epochs = 30
batch_size = 16

nb_train_samples = 16189

nb_validate_samples = 9909

nb_test_samples = 5747

def lr_decay(epoch):
    initial_lr = 1e-3
    drop = 0.9
    epochs_drop = 3
    lr = initial_lr * math.pow(drop, math.floor(epoch / epochs_drop))

    return lr
def my_model():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(time_step, 28, 28, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(1, kernel_size=(1, 1), padding='same', return_sequences=True, activation='sigmoid'))

    return seq
def train_model():
    seq = my_model()
    seq.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae', 'acc'])#'mean_squared_error'custom_objective

    mycallback = []

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
    model_checkpoint = ModelCheckpoint(
            '/home/lfan/Dropbox/runCoAtt/new_experiment/' + version + '/checkpoint.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            save_best_only=False,
            save_weights_only=True, monitor='val_loss')


    lrscheduler = LearningRateScheduler(lr_decay)

    mycallback.append(early_stopping)
    mycallback.append(model_checkpoint)
    mycallback.append(lrscheduler)

    print('Start Fitting')
    seq.fit_generator(my_generator(1, batch_size=30),
                          steps_per_epoch=nb_train_samples // 5  // batch_size,
                          epochs=epochs,
                          validation_data=my_generator(2, batch_size=15),
                          validation_steps=nb_validate_samples // 5 // batch_size,
                          callbacks=mycallback)

    seq.save_weights(model_weights_path)
    seq.save(model_path)
    score = seq.evaluate_generator(my_generator(2,batch_size=15), nb_validate_samples//5//time_step // batch_size)
    print("Validation Accuracy= ", score[1])


