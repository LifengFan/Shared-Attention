from keras.utils.data_utils import get_file
import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
from random import shuffle
import math
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.models import Model
from keras import optimizers
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, BatchNormalization, Activation, Dropout, Embedding, LSTM, Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten, concatenate, Reshape, Lambda
from keras.applications.vgg16 import VGG16
import scipy.ndimage as ndimage

import operator
from keras.applications.vgg16 import preprocess_input


# annot_path='/home/lfan/Dropbox/JointAttention/Data/annotation/'
#
# train_path='/home/lfan/Dropbox/JointAttention/faces/train_union/'
# train_sf=[f for f in listdir(train_path) if isdir(join(train_path,f))]
# validate_path='/home/lfan/Dropbox/JointAttention/faces/validate_union/'
# validate_sf=[f for f in listdir(validate_path) if isdir(join(validate_path,f))]
# test_path='/home/lfan/Dropbox/runCoAtt/rawData/images/separate/test/'
# test_sf=[f for f in listdir(test_path) if isdir(join(test_path,f))]

#
# vid_set=test_sf #train_sf
#
# annot_sf=[join(annot_path,f) for f in listdir(annot_path) if isdir(join(annot_path,f))]
#
# with open('/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_test_new.txt','w') as F:
#     for i in range(len(vid_set)):
#         sf = annot_path+vid_set[i]
#         vid = vid_set[i]
#
#         with open(join(sf, 'coattention.txt'), 'r') as r1:
#             lines = r1.readlines()
#         for j in range(0,len(lines),10):
#             list_now = lines[j].split()
#             frame_now = str(int(list_now[1])+1)
#             img_name = '/home/lfan/Dropbox/runCoAtt/rawData/images/all/' + frame_now.zfill(5) + '_' + vid + '.jpg'
#
#             ca_xmin=float(list_now[2])
#             ca_ymin=float(list_now[3])
#             ca_xmax=float(list_now[4])
#             ca_ymax=float(list_now[5])
#             ca_x=(ca_xmin+ca_xmax)/2
#             ca_y=(ca_ymin+ca_ymax)/2
#
#
#             num_face = (len(list_now) - 2) / 4 - 1
#             for k in range(num_face):
#                 face = list_now[(6 + k * 4):(10 + k * 4)]
#                 xmin=float(face[0])
#                 ymin=float(face[1])
#                 xmax=float(face[2])
#                 ymax=float(face[3])
#                 face_x=(xmin+xmax)/2
#                 face_y=(ymin+ymax)/2
#
#                 dir_x=ca_x-face_x
#                 dir_y=ca_y-face_y
#                 L=math.sqrt(dir_x ** 2 + dir_y ** 2)
#                 dir_x=dir_x/L
#                 dir_y=dir_y/L
#
#                 # if dir_y >= 0:
#                 #     direction = math.acos(dir_x / math.sqrt(dir_x ** 2 + dir_y ** 2))
#                 # elif dir_y < 0:
#                 #     direction = 2*math.pi - math.acos(dir_x / math.sqrt(dir_x ** 2 + dir_y ** 2))
#
#                 F.write(img_name+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' '+str(dir_x)+' '+str(dir_y)+'\n')


# with open('/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_train.txt','r') as f:
#     lines=f.readlines()
#
# for i in range(len(lines)):
#     list=lines[i].split()
#     if not isfile(list[0]):
#
#         print(list[0])

# with open('/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_train.txt','r') as f:
#     lines=f.readlines()
#
# dir_hist=np.zeros(shape=(10,1))
# for i in range(len(lines)):
#     list=lines[i].split()
#
#     dir=int(float(list[5])//(2*math.pi/10))
#     if dir==10:
#         dir=0
#     dir_hist[dir,0]+=1
#
# print(dir_hist)

# with open('/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_test_new.txt','r') as f:
#     lines=f.readlines()
# with open('/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_test_flipped_new.txt','w') as f2:
#     for i in range(len(lines)):
#         list = lines[i].split()
#         xmin=float(list[1])
#         xmax=float(list[3])
#
#         xmax_n=480-xmin
#         xmin_n=480-xmax
#
#         dir_x=float(list[5])
#         dir_y=float(list[6])
#
#         dir_x=-dir_x
#         # if dir<0.5*(2*math.pi):
#         #     dir_n=0.5-dir
#         # else:
#         #     dir_n=1.5*(2*math.pi)-dir
#
#         f2.write(list[0]+' '+str(xmin_n)+' '+list[2]+' '+str(xmax_n)+' '+list[4]+' '+str(dir_x)+' '+str(dir_y)+' '+'f\n')


# gfdatapath='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/'
#
# with open('/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_test.txt','r') as f:
#     lines=f.readlines()
# with open('/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_test_flipped.txt','w') as f2:
#     for i in range(len(lines)):
#         list = lines[i].split()
#         img=cv2.imread(join(gfdatapath,list[0]))
#         h, w, ch = img.shape
#
#         xmin=float(list[1])
#         xmax=float(list[3])+float(list[1])
#
#         xmax_n=w-xmin
#         xmin_n=w-xmax
#         w_n=xmax_n-xmin_n
#
#         dir=float(list[9])
#         if dir<0.5:
#             dir_n=0.5-dir
#         else:
#             dir_n=1.5-dir
#
#         f2.write(list[0]+' '+str(xmin_n)+' '+list[2]+' '+str(w_n)+' '+list[4]+' '+list[5]+' '+list[6]+' '+list[7]+' '+list[8]+' '+str(dir_n)+' '+'f\n')


batch_size=25
epochs=25

nb_train_samples=12000 # 13706
nb_validate_samples=6000 #8533
nb_test_samples=6000

model_weights_path = '/home/lfan/Dropbox/runCoAtt/new_experiment/gazedir/gazedir_finalweights.hdf5'
model_path = '/home/lfan/Dropbox/runCoAtt/new_experiment/gazedir/gazedir_finalmodel.h5'
tensorboard_log_dir = '/home/lfan/Dropbox/runCoAtt/new_experiment/gazedir/tb_log/'

gfdatapath='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/'


def mygenerator(mode):

    if mode==1:
        file_name='/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_train_new.txt'
        #file_name_flip='/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_train_flipped.txt'
        #file_name_gazefollow='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_train.txt'
        #file_name_gazefollow_flipped = '/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_train_flipped.txt'
        sample_len=nb_train_samples
    elif mode==2:
        file_name = '/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_validate_new.txt'
        #file_name_flip = '/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_validate_flipped.txt'
        #file_name_gazefollow = '/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_test.txt'
        #file_name_gazefollow_flipped = '/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/annotation_test_flipped.txt'
        sample_len=nb_validate_samples

    with open(file_name,'r') as reader:
        lines=reader.readlines()
    # with open(file_name_flip,'r') as reader:
    #     lines_flipped=reader.readlines()
    # with open(file_name_gazefollow,'r') as reader:
    #     lines_gazefollow=reader.readlines()
    # with open(file_name_gazefollow_flipped,'r') as reader:
    #     lines_gazefollow_flipped=reader.readlines()

    lines=lines[0:sample_len]
    #lines_flipped=lines_flipped[0:sample_len]

    #lines[0:0]=lines_flipped
    # lines[0:0]=lines_gazefollow
    # lines[0:0]=lines_gazefollow_flipped

    #lines=lines[0:sample_len]

    shuffle(lines)
    shuffle(lines)


    cur_batch_index=0
    while True:
        x_batch = np.zeros(shape=(batch_size, 224, 224, 3))
        y_batch = np.zeros(shape=(batch_size, 2))

        start_id = cur_batch_index * batch_size
        end_id = start_id + batch_size
        files_batch_now = lines[start_id:end_id]

        for j in range(batch_size):
            list_now=files_batch_now[j].split()

            if len(list_now)<8:

                img = cv2.imread(list_now[0])
                # if len(list_now) == 7:
                #     img = cv2.flip(img, 1)

                # cv2.imshow('flipped',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                xmin = int(float(list_now[1]))# - (float(list_now[3]) - float(list_now[1])) * 0)
                ymin = int(float(list_now[2]))# - (float(list_now[4]) - float(list_now[2])) * 0)
                xmax = int(float(list_now[3]))# + (float(list_now[3]) - float(list_now[1])) * 0)
                ymax = int(float(list_now[4]) )#+ (float(list_now[4]) - float(list_now[2])) * 0)

                # w = float(xmax - xmin)
                # h = float(ymax - ymin)
                # wh_list = [w, h]
                # max_index, max_value = max(enumerate(wh_list), key=operator.itemgetter(1))
                # pad = (wh_list[max_index] - wh_list[1 - max_index]) / 2
                # if max_index == 0:
                #     ymin = ymin - pad
                #     ymax = ymax + pad
                # elif max_index == 1:
                #     xmin -= pad
                #     xmax += pad

                xmin = int(max(0, xmin))
                ymin = int(max(0, ymin))

                xmax = max(xmin + 1, xmax)
                ymax = max(ymin + 1, ymax)

                xmax = int(min(479, xmax))
                ymax = int(min(319, ymax))

                #direction = float(list_now[5]) / (2 * math.pi)
                dir_x=float(list_now[5])
                dir_y=float(list_now[6])
                # print(img.shape)
                face = img[ymin:ymax, xmin:xmax, :]
                face = cv2.resize(face, (224, 224))

                #face=face.astype('float32')
                # face[:, :, 0] -= 123.68
                # face[:, :, 1] -= 116.779
                # face[:, :, 2] -= 103.939
                #face=(face/255)*2-1
                #print(direction)
                #cv2.putText(face, str(direction), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.imshow('face',face)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                x_batch[j, :, :, :] = face
                y_batch[j, 0] = dir_x
                y_batch[j, 1] = dir_y

            elif len(list_now)>9:

                name = list_now[0]
                bbox_x = int(float(list_now[1]))
                bbox_y = int(float(list_now[2]))
                bbox_w = int(float(list_now[3]))
                bbox_h = int(float(list_now[4]))
                eye_x = int(np.float32(list_now[5]))
                eye_y = int(np.float32(list_now[6]))
                direction = np.float32(list_now[9])

                img = cv2.imread(gfdatapath + name)
                if len(list_now)==11:
                    img = cv2.flip(img, 1)

                h, w, ch = img.shape
                totop = np.abs(eye_y - bbox_y)
                if int(totop) == 0:
                    totop = 10
                face_h = int(2* totop)
                face_w = int(2 * totop)
                face_x = int(eye_x - totop)
                face_y = int(eye_y - totop)

                if face_x < bbox_x:
                    face_x = bbox_x
                if face_y < bbox_y:
                    face_y = bbox_y
                if (face_x + face_w) > (bbox_x + bbox_w):
                    face_w = bbox_x + bbox_w - face_x
                if (face_y + face_h) > (bbox_y + bbox_h):
                    face_h = bbox_y + bbox_h - face_y

                if face_x < 0:
                    face_x = 0
                if face_y < 0:
                    face_y = 0

                face_w=max(1,face_w)
                face_h=max(1,face_h)

                if (face_x + face_w) > w:
                    face_w = w - face_x
                if (face_y + face_h) > h:
                    face_h = h - face_y

                face_pro = img[face_y:(face_y + face_h), face_x:(face_x + face_w), :]
                face_pro = cv2.resize(face_pro, (224, 224))

                face_pro=face_pro.astype('float32')
                # face_pro[:, :, 0] -= 103.939
                # face_pro[:, :, 1] -= 116.779
                # face_pro[:, :, 2] -= 123.68

                face_pro = np.float32(face_pro) / 255
                #direction = direction / (math.pi * 2)

                x_batch[j,:,:,:] = face_pro
                y_batch[j,0] = direction

            # cv2.imshow(str(direction),x_batch[j,:,:,:])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        yield  x_batch, y_batch

        cur_batch_index = cur_batch_index + 1
        if cur_batch_index >= (len(lines) // batch_size):
            cur_batch_index = 0
            shuffle(lines)
#
def myloss(y_true,y_pred):

    lamb=0.9
    loss=lamb*tf.reduce_sum(tf.square(y_true-y_pred))
    sess=tf.Session()
    sess.run(loss)
    #+(1-lamb)*tf.abs(1-tf.reduce_sum(tf.square(y_pred)))
    #loss= (1 - tf.matmul(y_true,y_pred,transpose_b=True)/tf.sqrt(tf.reduce_sum(tf.square(y_true))*tf.reduce_sum(tf.square(y_pred)))) #+ tf.abs(1-tf.reduce_sum(tf.square(y_pred)))
    # offset=np.abs(y_true-y_pred)
    # loss=2*((0.5-np.abs(offset-0.5))**2)

    return loss


def lr_decay(epoch):
    initial_lr = 1e-3
    drop = 0.8
    epochs_drop = 2
    lr = initial_lr * math.pow(drop, math.floor(epoch / epochs_drop))

    return lr

def train_model():

    img_input = Input(shape=(224, 224, 3), name='img_input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    base_output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    base_model=Model(inputs=img_input,outputs=base_output)
    base_model.trainable=True
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')

    base_model.load_weights(weights_path)

    output = Flatten(name='flatten')(base_output)
    output = Dense(4096, kernel_initializer='normal',activation='relu',name='fc1')(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, kernel_initializer='normal',name='fc2')(output)
    output=BatchNormalization()(output)
    output=Activation('relu')(output)

    output = Dropout(0.5)(output)
    predict= Dense(2, kernel_initializer='normal',activation='tanh', name='predict')(output)

    # Create your own model
    mymodel = Model(inputs=img_input, outputs=predict)
    mymodel.summary()

    mycallback = []
    tbCallback = TensorBoard(log_dir=tensorboard_log_dir,
                             histogram_freq=0, batch_size=batch_size,
                             write_graph=True, write_images=True)
    lrscheduler = LearningRateScheduler(lr_decay)
    model_checkpoint = ModelCheckpoint(
        '/home/lfan/Dropbox/runCoAtt/new_experiment/gazedir/checkpoint.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=False,
        save_weights_only=True, monitor='val_loss')

    mycallback.append(tbCallback)
    mycallback.append(model_checkpoint)
    mycallback.append(lrscheduler)
    #sgd = optimizers.SGD(lr=1e-4)
    mymodel.compile(optimizer='sgd', loss='mse', metrics=['mae','acc'])

    history = mymodel.fit_generator(mygenerator(1),
                                     steps_per_epoch=(2*nb_train_samples) // batch_size,
                                     epochs=epochs,
                                     validation_data=mygenerator(2),
                                     validation_steps=(2*nb_validate_samples) // batch_size,
                                     callbacks=mycallback)
    mymodel.save_weights(model_weights_path)
    mymodel.save(model_path)
    score = mymodel.evaluate_generator(mygenerator(2), (2*nb_validate_samples) // batch_size)
    print("Validation Accuracy= ", score[1])

def testmodel():

    # img_input = Input(shape=(224, 224, 3), name='img_input')
    #
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
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # base_output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #
    # output = Flatten(name='flatten')(base_output)
    # output = Dense(1024, activation='tanh', name='fc1')(output)
    # output = Dropout(0.5)(output)
    # output = Dense(512, activation='tanh', name='fc2')(output)
    # output = Dropout(0.5)(output)
    # output = Dense(128, activation='tanh', name='fc3')(output)
    # output = Dropout(0.5)(output)
    # predict = Dense(2, activation='tanh', name='predict')(output)

    img_input = Input(shape=(224, 224, 3), name='img_input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    base_output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    output = Flatten(name='flatten')(base_output)
    output = Dense(4096, kernel_initializer='normal', activation='relu', name='fc1')(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, kernel_initializer='normal', name='fc2')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(0.5)(output)
    predict = Dense(2, kernel_initializer='normal', activation='tanh', name='predict')(output)

    # Create your own model
    mymodel = Model(inputs=img_input, outputs=predict)
    mymodel.summary()

    mymodel.load_weights('/home/lfan/Dropbox/runCoAtt/new_experiment/gazedir/checkpoint.weights.19-0.47.hdf5')

    file_name='/home/lfan/Dropbox/runCoAtt/rawData/gaze_summary_test_new.txt'
    sample_len=nb_test_samples

    with open(file_name,'r') as reader:
        lines=reader.readlines()

    lines=lines[0:sample_len]
    shuffle(lines)
    LOSS=0
    cnt=0
    for j in range(sample_len):
        list_now = lines[j].split()

        img = cv2.imread(list_now[0])
        xmin = int(float(list_now[1])) # - (float(list_now[3]) - float(list_now[1])) * 0)
        ymin = int(float(list_now[2])) # - (float(list_now[4]) - float(list_now[2])) * 0)
        xmax = int(float(list_now[3])) # + (float(list_now[3]) - float(list_now[1])) * 0)
        ymax = int(float(list_now[4])) # + (float(list_now[4]) - float(list_now[2])) * 0)

        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = max(xmin + 1, xmax)
        ymax = max(ymin + 1, ymax)

        xmax = min(479, xmax)
        ymax = min(319, ymax)

        dir_x=float(list_now[5])
        dir_y=float(list_now[6])

        #direction = float(list_now[5]) / (2 * math.pi)

        # print(img.shape)
        face = img[ymin:ymax, xmin:xmax, :]
        face = cv2.resize(face, (224, 224))

        #face = face.astype('float32')
        #face=face[:,:,::-1]

        # face[:, :, 0] -= 123.68
        # face[:, :, 1] -= 116.779
        # face[:, :, 2] -= 103.939

        #face=face/255
        # print(direction)
        # cv2.putText(face, str(direction), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

        x_batch=np.zeros(shape=(1,224,224,3))
        x_batch[0,:,:,:]=face
        res=mymodel.predict(x_batch,batch_size=1)
        res=res[0]
        print(res)
        print('GT: '+str(dir_x)+' '+str(dir_y))

        o_x=112
        o_y=112

        cv2.arrowedLine(face,(o_x,o_y),(o_x+int(res[0]*100),o_y+int(res[1]*100)),(0,0,255),5)
        cv2.arrowedLine(face, (o_x, o_y), (o_x + int(dir_x*100), o_y + int(dir_y*100)), (0, 255, 255), 5)
        cv2.imshow('face',face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        LOSS+=((dir_x-res[0])**2+(dir_y-res[1])**2)/2
        cnt+=1
        # cv2.imshow(str(res[0])+' '+str(res[1]), face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print('Average loss: {}'.format(LOSS/cnt))


#train_model()
testmodel()
