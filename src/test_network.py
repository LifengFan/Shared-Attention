from generator import my_generator
import math
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from gazemap import getGazeHeatmap
from os.path import isfile, join, isdir
import numpy as np
import cv2
from model_generator import my_model
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation,concatenate
from scipy.ndimage.filters import gaussian_filter
from random import shuffle
import matplotlib.pyplot as plt
from roc_metric import eval_bbx
from distance_metric import getDistance, getDistance1


batch_size = 25
epochs = 40

nb_train_pos = 7000
nb_train_neg = 2000
nb_train_samples = 9000

nb_validate_pos = 4500
nb_validate_neg = 1200
nb_validate_samples = 5700

nb_test_pos = 2522
nb_test_neg = 2522
nb_test_samples = nb_test_pos + nb_test_neg

RANGE = 1000
DIS_RANGE = 50

version = 'method8'
model_weights_path = '~/Projects/CVPR2018/NN/experiment/' + version + '/finalweights.hdf5'
model_path = '~/Projects/CVPR2018/NN/experiment/' + version + '/finalmodel.h5'
tensorboard_log_dir = '~/Projects/CVPR2018/NN/experiment/' + version + '/tb_log/'
prop_path = '/media/yixin/Elements/bbx_proposal/'

images_train_path = '/media/yixin/Elements/CM_DATA/indoor/cm/'
images_validate_path = '/media/yixin/Elements/CM_DATA/indoor/cm/'
images_test_path = '/media/yixin/Elements/CM_DATA/indoor/cm/'

def get_test_res_all():


    mymodel = my_model()
    mymodel.load_weights(
        '/home/yixin/Projects/CVPR2018/NN/experiment/' + version + '/checkpoint.weights.30-0.15.hdf5')

    ## prepare testing data
    file_name_pos = 'test_coatt_summary_pos.txt'
    file_name_neg = 'test_coatt_summary_neg.txt'
    gaze_dir_prefix = '/home/yixin/Dropbox/JointAttention/tested_face_direction/test/'


    with open(file_name_pos, 'r') as data_to_read:
        data_lines_pos = data_to_read.readlines()
    with open(file_name_neg, 'r') as data_to_read:
        data_lines_neg = data_to_read.readlines()

    # shuffle(data_lines_pos)
    # shuffle(data_lines_neg)

    data_lines_pos = data_lines_pos[0:nb_test_pos]
    data_lines_neg = data_lines_neg[0:nb_test_neg]

    data_lines_pos[0:0] = data_lines_neg
    data_lines = data_lines_pos
    # shuffle(data_lines)

    ifframe_s = np.zeros((1,RANGE))
    tp = np.zeros((1,RANGE))
    fp = np.zeros((1, RANGE))
    true_pos_s = np.zeros((1,RANGE))
    detected_bbx_s = np.ones((1,RANGE))
    num_ca_s = 0#np.zeros((1,RANGE))
    # distance = np.zeros((1, DIS_RANGE))
    distance = 0

    pos = 0
    neg = 0

    for test_id in range(len(data_lines)):
        print(test_id)

        list_now = data_lines[test_id].split()
        vid = list_now[0].split('_')[-1][:-4]

        imgfilename = images_test_path + list_now[0].split('/')[-1][6:-4] + '/' + str(int(list_now[0].split('/')[-1][0:5])) + '.jpg'
        if isfile(imgfilename) == False:
            print(imgfilename)
        img = cv2.imread(imgfilename)
        img_orig = img

        x_batch = np.zeros(shape=(1, 224, 224, 3))

        sum_thres = 8000
        crop_margin = 0

        if np.sum(img[0:40, :, :]) < sum_thres:
            img = img[40:-40, :, :]
            crop_margin = 40
        elif np.sum(img[0:30, :, :]) < sum_thres:
            img = img[30:-30, :, :]
            crop_margin = 30
        elif np.sum(img[0:25, :, :]) < sum_thres:
            img = img[25:-25, :, :]
            crop_margin = 25
        elif np.sum(img[0:20, :, :]) < sum_thres:
            img = img[20:-20, :, :]
            crop_margin = 20

        img = cv2.resize(img, (224, 224))
        x_batch[0, :, :, :] = img


        ### gaze heatmap part

        # gaze_heatmap_batch = np.zeros(shape=(1, 224, 224, 1))
        gaze_heatmap_batch = np.zeros(shape=(1, 28, 28, 1))
        if isfile(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt'):
            with open(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt', 'r') as gaze_to_read:
                gaze_lines = gaze_to_read.readlines()

            gazeheatmap = np.zeros(shape=(320, 480))

            for gaze_id in range(len(gaze_lines)):
                gaze_list = gaze_lines[gaze_id].split()

                g_xmin = float(gaze_list[3]) * (24.0 / 480)
                g_ymin = (float(gaze_list[4])) * (16.0 / 320)
                g_xmax = float(gaze_list[5]) * (24.0 / 480)
                g_ymax = (float(gaze_list[6])) * (16.0 / 320)

                g_x = (g_xmin + g_xmax) / 2
                g_y = (g_ymin + g_ymax) / 2

                dir_x = float(gaze_list[7])
                dir_y = float(gaze_list[8])

                gazehp = getGazeHeatmap([g_x, g_y], [dir_x, dir_y])
                gazehp = cv2.resize(gazehp, (480, 320))
                gazeheatmap += gazehp

            if crop_margin > 0:
                gazeheatmap = gazeheatmap[crop_margin:-crop_margin, :]

            gaze_heatmap_batch[0, :, :, 0] = cv2.resize(gazeheatmap, (28, 28))
            # gaze_heatmap_batch[0, :, :, 0] = cv2.resize(gazeheatmap, (224, 224))

        ### face bbx part
        prop_batch = np.zeros(shape=(1, 28, 28, 1))
        # prop_batch = np.zeros(shape=(1, 224, 224, 1))
        face_tmp = np.zeros(shape=(320, 480))
        if isfile(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt'):
            with open(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt', 'r') as gaze_to_read:
                gaze_lines = gaze_to_read.readlines()

            for gaze_id in range(len(gaze_lines)):
                gaze_list = gaze_lines[gaze_id].split()

                # cv2.rectangle(img_orig, (int(gaze_list[3]), int(gaze_list[4])),
                #               (int(gaze_list[5]), int(gaze_list[6])), (0, 255, 255))

                g_xmin = int(float(gaze_list[3]))
                g_ymin = int(float(gaze_list[4]))
                g_xmax = int(float(gaze_list[5]))
                g_ymax = int(float(gaze_list[6]))

                face_tmp[g_ymin:g_ymax, g_xmin:g_xmax] = 1

            # if crop_margin > 0:
            #     face_tmp = face_tmp[crop_margin:-crop_margin, :]

        ### object proposal bbx part

        if isfile(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt'):

            # object_tmp = np.zeros(shape=(320, 480))

            with open(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt',
                      'r') as f2:
                while True:
                    line = f2.readline()
                    if len(line) == 0:
                        break
                    list = line.split()
                    xmin = float(list[0])
                    ymin = float(list[1])
                    xmax = float(list[2])
                    ymax = float(list[3])

                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)

                    face_tmp[ymin:ymax, xmin:xmax] = 1
                    face_ori = face_tmp

        if crop_margin > 0:
            face_tmp = face_tmp[crop_margin:-crop_margin, :]
        prop_batch[0, :, :, 0] = cv2.resize(face_tmp, (28, 28))
                # prop_batch[0, :, :, 0] = cv2.resize(face_tmp, (224, 224))
        ### add gaussian filter to prop heatmap
        # prop_batch[0, :, :, :] = gaussian_filter(prop_batch[0, :, :, :], 2)


        ### gt part
        y_batch=np.zeros(shape=(1,28,28,1))
        y_ori = np.zeros(shape=(320, 480))
        if len(list_now) > 1:

            y_tmp = np.zeros(shape=(320, 480))

            num_ca = (len(list_now) - 1) / 4
            for ca_id in range(num_ca):
                xmin = float(list_now[4 * ca_id + 1])
                ymin = float(list_now[4 * ca_id + 2])
                xmax = float(list_now[4 * ca_id + 3])
                ymax = float(list_now[4 * ca_id + 4])

                cv2.rectangle(img_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)

                y_tmp[ymin:ymax, xmin:xmax] = 1

            y_ori = y_tmp

            if crop_margin > 0:
                y_tmp = y_tmp[crop_margin:-crop_margin, :]

            y_batch[0,:,:,0] = cv2.resize(y_tmp, (28, 28))

        res = mymodel.predict([prop_batch], batch_size=1)
        # res = mymodel.predict([x_batch, gaze_heatmap_batch, prop_batch], batch_size=1)

        res = res[0][:, :, 0]

        y_res = np.zeros(shape=(320, 480))
        tt = cv2.resize(res, (480,(320 - 2 * crop_margin)))
        y_res[crop_margin:(320-crop_margin), :] = tt

        [tp_t, fp_t, ifframe, true_pos, detected_bbx, num_ca] = eval_bbx(y_res,list_now)
        dis = getDistance1(y_res, y_ori, 0.12)

        ifframe_s = ifframe_s + ifframe
        true_pos_s = true_pos_s + true_pos
        detected_bbx_s = detected_bbx_s + detected_bbx
        num_ca_s = num_ca_s + num_ca
        distance = distance + dis
        tp = tp + tp_t
        fp = fp + fp_t
        if num_ca > 0:
            pos = pos + 1
        else:
            neg = neg + 1

        # plt.figure('test result for image {}'.format(list_now[0].split('/')[-1]))
        #
        # plt.subplot(2, 3, 1)
        # plt.title('original image')
        # plt.imshow(img)  # +y_batch[j, :, :, :]/20)
        #
        # plt.subplot(2, 3, 2)
        # plt.title('gazeheatmap')
        # plt.imshow(y_ori)
        #
        # # plt.subplot(2, 3, 2)
        # # plt.title('gazeheatmap')
        # # plt.imshow(cv2.resize(gaze_heatmap_batch[0], (224, 224)))
        #
        # plt.subplot(2, 3, 3)
        # plt.title('propheatmap')
        # plt.imshow(y_res)
        #
        # # plt.subplot(2, 3, 3)
        # # plt.title('propheatmap')
        # # plt.imshow(cv2.resize(prop_batch[0], (224, 224)))
        #
        # plt.subplot(2, 3, 4)
        # plt.title('gt coatt')
        # plt.imshow(cv2.resize(y_batch[0], (224, 224)))
        #
        # plt.subplot(2, 3, 5)
        # plt.title('res coatt')
        # plt.imshow(cv2.resize(res, (224, 224)))
        #
        # plt.subplot(2, 3, 6)
        # plt.title('propheatmap')
        # plt.imshow(face_ori)
        #
        # # g = lambda x: np.multiply(x[0], x[1])
        # # plt.subplot(2, 3, 6)
        # # plt.title('multiply')
        # # plt.imshow(cv2.resize(g([gaze_heatmap_batch[0], prop_batch[0]]), (224, 224)))
        #
        # plt.axis('off')
        # plt.show()
    plt.figure()
    # print(sum(num_ca_s))
    ifframe_s = ifframe_s / float(nb_test_samples)
    recall = true_pos_s / float(num_ca_s)
    # dis_avg = distance / float(num_ca_s)
    precision = true_pos_s / detected_bbx_s
    filename = '/home/yixin/Projects/CVPR2018/NN/prc/' + version +'precision'
    np.save(filename,precision)
    filename = '/home/yixin/Projects/CVPR2018/NN/prc/' + version + 'recall'
    np.save(filename, recall)
    x_axis = np.linspace(1 / RANGE, 1, RANGE)
    x_axis = np.reshape(x_axis, (1, RANGE))

    # dis_x_axis = np.linspace(1 / DIS_RANGE, 1, DIS_RANGE)
    # dis_x_axis = np.reshape(dis_x_axis, (1, DIS_RANGE))

    tpr = tp / float(pos)
    fpr = fp / float(neg)

    plt.subplot(1, 4, 1)
    plt.title('frame acc')
    plt.plot(x_axis, ifframe_s, 'ro')
    plt.subplot(1, 4, 2)
    plt.title('prec_recall')
    plt.plot(recall, precision, 'ro')
    plt.subplot(1, 4, 3)
    plt.title('distance')
    # plt.plot(dis_x_axis, dis_avg, 'ro')
    plt.subplot(1, 4, 4)
    plt.title('roc')
    plt.plot(tpr, fpr, 'ro')
    plt.show()

    print(np.max(ifframe_s))
    # print(np.argmin(dis_avg))
    print(distance/float(num_ca_s))


get_test_res_all()
# score = np.array([1,2,3,4,5,6])
# judge = np.where(score >=  3)
# print(len(judge))
# judge = judge[0]
# print(len(judge))
# x_axis = np.zeros((1,1001))
# x_axis[0,:] = np.linspace(0,1,RANGE)
# x_axis = np.asarray(x_axis[0])
# print(x_axis.shape())

# a = [[1,2],[3,4]]
# a = np.asarray(a)
# b = np.array([1,2])
# print(np.delete(a,'0',axis=0))
# list_now =['/home/lfan/Dropbox/runCoAtt/rawData/images/separate/test/21/00501_21.jpg']
# imgfilename = images_test_path + list_now[0].split('/')[-1][6:-4] + '/' + str(int(list_now[0].split('/')[-1][0:5])) + '.jpg'
# img = cv2.imread(imgfilename)
# img_orig = img



