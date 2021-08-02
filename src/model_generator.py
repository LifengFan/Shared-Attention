import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation,concatenate


def my_model( ):
    # img_input = Input(shape=(224, 224, 3), name='img_input')
    # gaze_hp_input = Input(shape=(28, 28, 1), name='gaze_input')
    prop_input = Input(shape=(28, 28, 1), name='prop_input')

    # # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #
    # # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #
    # # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #
    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='prediction1')(x)

    # gaze = Conv2D(32, (3, 3), activation='relu', padding='same', name='block6_conv1')(gaze_hp_input)
    # gaze = Conv2D(16, (3, 3), activation='relu', padding='same', name='block6_conv2')(gaze)
    # gaze = Conv2D(8, (3, 3), activation='relu', padding='same', name='block6_conv3')(gaze)
    # gaze = Conv2D(1, (1, 1), activation='relu', padding='same', name='block6_conv4')(gaze)
    #
    #
    # prop = Conv2D(32, (3, 3), activation='relu', padding='same', name='block7_conv1')(prop_input)
    # prop = Conv2D(16, (3, 3), activation='relu', padding='same', name='block7_conv2')(prop)
    # prop = Conv2D(8, (3, 3), activation='relu', padding='same', name='block7_conv3')(prop)
    # prop = Conv2D(1, (1, 1), activation='relu', padding='same', name='block7_conv4')(prop)

    # g = lambda x: np.multiply(x[0], x[1])

    # output = concatenate([prop_input, gaze_hp_input], axis=3)

    # output = Lambda(g, output_shape=(28, 28, 1))([gaze, prop])
    output = Conv2D(16, (3, 3), padding='same', name='block8_conv1')(prop_input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Conv2D(16, (3, 3), padding='same', name='block8_conv2')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Conv2D(8, (3, 3), padding='same', name='block8_conv3')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    # output = Conv2D(4, (3, 3), padding='same', name='block8_conv4')(output)
    # output = BatchNormalization()(output)
    # output = Activation('relu')(output)
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='prediction2')(output)

    mymodel = Model(inputs=[prop_input], outputs=output)
    mymodel.summary()
    mymodel.trainable = True


    return mymodel
