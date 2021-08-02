
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Activation, Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, Merge
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from MyBatchGenerator import TwoStream_generator
from keras.utils.visualize_util import plot
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from MyBatchGenerator import Img_From_List

import VGGCAM
import convnets

nb_classes = 22

train_data_x = '/home/yang/Desktop/ICCV2017/Data/Path/FullImgObj_Training_X.txt'
train_data_y = '/home/yang/Desktop/ICCV2017/Data/Path/FullImgObj_Training_Y.txt'


validation_data_x = '/home/yang/Desktop/ICCV2017/Data/Path/FullImgObj_Validation_X.txt'
validation_data_y = '/home/yang/Desktop/ICCV2017/Data/Path/FullImgObj_Validation_Y.txt'

batch_size = 64

# Get trainnum and validation num
train_file = np.genfromtxt(train_data_y)
nb_train = len(train_file)
validation_file = np.genfromtxt(validation_data_y)
nb_validation = len(validation_file)

# VGG Model
model = convnets.convnet('vgg_16', weights_path= 'weights/vgg16_weights_FullImg22Classes.h5', heatmap=False, nb_classes=22)

for layer in model.layers[:25]:    # freezing top 25 layers weight
    layer.trainable = False
# model.pop()
# model.pop()
# model.add(Dense(nb_classes, name='dense_3'))
# model.add(Activation("softmax",name="softmax"))

model.compile(optimizer=SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = Img_From_List(batch_size,train_data_x,train_data_y,nb_classes)
validation_generator = Img_From_List(batch_size,validation_data_x,validation_data_y,nb_classes)

early_stopping = EarlyStopping(verbose=1, patience=30, monitor='acc')
model_checkpoint = ModelCheckpoint('/home/yang/Desktop/ICCV2017/Code/Keras/weights/vgg16_weights_FullImg22Classes_renew.h5', save_best_only=True, save_weights_only=True,
                                    monitor='acc')
callbacks_list = [early_stopping, model_checkpoint]

history = model.fit_generator(
        train_generator,
        samples_per_epoch=(nb_train//batch_size)*batch_size,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=(nb_validation//batch_size)*batch_size,
        callbacks=callbacks_list)

VGGCAM.save_history(history=history, prefix='FullImg-22classes-fine-tuning-renew', plots_dir='/home/yang/Desktop/ICCV2017/Code/Keras/plots/')

