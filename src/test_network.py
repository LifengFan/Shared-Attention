import numpy as np
from random import shuffle
import os
from os.path import isfile, join, isdir
import cv2
import math
import matplotlib.pyplot as plt
from lstm_method2 import my_model
from roc_metric import eval_bbx
from distance_metric import getDistance, getDistance1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

time_step = 10
version = 'lstm_2'
RANGE = 1000
DIS_RANGE = 50

file_name = '/home/lfan/Dropbox/JointAttention/Data/test_coatt_summary_contin.txt'
index_set = np.load('/home/lfan/Dropbox/JointAttention/Data/test_index.npy')

mymodel=my_model()
mymodel.load_weights('./checkpoint.weights.09-0.02.hdf5')

with open(file_name, 'r') as data_to_read:
    data_lines = data_to_read.readlines()

gaze_dir_prefix = '/home/lfan/Dropbox/JointAttention/tested_face_direction/test/'
prop_path = '/home/lfan/Dropbox/JointAttention/bbx proposal/'
shuffle(index_set)
#index_set = index_set[0:10]

# ifframe_s = np.zeros((1,RANGE))
# tp = np.zeros((1,RANGE))
# fp = np.zeros((1, RANGE))
# true_pos_s = np.zeros((1,RANGE))
# detected_bbx_s = np.ones((1,RANGE))
# num_ca_s = 0#np.zeros((1,RANGE))
# distance = np.zeros((1, DIS_RANGE))
# distance = 0

def getGazeHeatmap(face_loc, gaze_dir):
    sigma = 0.5
    heatmap = np.zeros(shape=(16, 24))
    dx = gaze_dir[0]
    dy = gaze_dir[1]
    fx = face_loc[0]
    fy = face_loc[1]

    for i in range(24):
        for j in range(16):
            theta = math.acos(
                ((i - fx) * dx + (j - fy) * dy) / math.sqrt((dx ** 2 + dy ** 2+0.00001) * ((i - fx) ** 2 + (j - fy) ** 2+0.00001)))
            heatmap[j, i] = (1. / sigma) * math.exp(-theta ** 2 / (2 * (sigma ** 2)))

    return heatmap

pos = 0
neg = 0
nb_test_samples = 0
#print(index_set)
for test_id in range(len(index_set)):
    print('test_id {}'.format(test_id))

    x_batch = np.zeros(shape=(1, time_step, 224, 224, 3))
    heatmap_batch = np.zeros(shape=(1, time_step, 28, 28, 1))
    y_batch = np.zeros(shape=(1, time_step, 28, 28, 1))
    y_ori = np.zeros(shape=(1, time_step, 320, 480, 1))
    face_prop_batch = np.zeros(shape=(1, time_step, 28, 28, 1))
    gaze_heatmap_batch = np.zeros(shape=(1, time_step, 28, 28, 1))
    prop_batch = np.zeros(shape=(1, time_step, 28, 28, 1))
    crop_batch = np.zeros(shape=(1, time_step))
    index_start = index_set[test_id]
    print('index_start {}'.format(index_start))

    for t in range(time_step):
        nb_test_samples = nb_test_samples + 1
        #list_now = data_lines[index_start * time_step + t].split()
        list_now = data_lines[index_start * 10 + t].split()
        vid = list_now[0].split('_')[-1][:-4]

        ### image part

        img = cv2.imread(list_now[0])
        img_orig = img


        # cv2.imshow('single img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


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

        crop_batch[0,t] = crop_margin

        img = cv2.resize(img, (224, 224))

        x_batch[0, t, :, :, :] = img

        ### gaze heatmap part


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
            # gaze_heatmap_batch[j, :, :,0] = cv2.resize(gazeheatmap,(224,224))
            gaze_heatmap_batch[0, t, :, :, 0] = cv2.resize(gazeheatmap, (28, 28))

            ### face bbx part

            if isfile(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt'):
                with open(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt', 'r') as gaze_to_read:
                    gaze_lines = gaze_to_read.readlines()

                face_tmp = np.zeros(shape=(320, 480))

                for gaze_id in range(len(gaze_lines)):
                    gaze_list = gaze_lines[gaze_id].split()

                    # cv2.rectangle(img_orig, (int(gaze_list[3]), int(gaze_list[4])),
                    #               (int(gaze_list[5]), int(gaze_list[6])), (0, 255, 255))

                    g_xmin = int(float(gaze_list[3]))
                    g_ymin = int(float(gaze_list[4]))
                    g_xmax = int(float(gaze_list[5]))
                    g_ymax = int(float(gaze_list[6]))

                    face_tmp[g_ymin:g_ymax, g_xmin:g_xmax] = 1

                if crop_margin > 0:
                    face_tmp = face_tmp[crop_margin:-crop_margin, :]

                face_prop_batch[0, t, :, :, 0] = cv2.resize(face_tmp, (28, 28))

                # cv2.imshow('orig img with face bbx',img_orig)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # cv2.imshow('face prop',prop_batch[j])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            ### object proposal bbx part


            if isfile(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt'):

                object_tmp = np.zeros(shape=(320, 480))

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

                        object_tmp[ymin:ymax, xmin:xmax] = 1

                    if crop_margin > 0:
                        object_tmp = object_tmp[crop_margin:-crop_margin, :]
                    # prop_batch[j,:,:,0] = cv2.resize(face_tmp,(224,224))
                    prop_batch[0, t, :, :, 0] = cv2.resize(object_tmp, (28, 28))

            ### add gaussian filter to prop heatmap

            # record=cv2.resize(object_tmp,(224,224))
            prop_batch[0, t, :, :, :] = face_prop_batch[0, t, :, :, :] + prop_batch[0, t, :, :, :]
            prop_batch[0, t][prop_batch[0, t] > 1] = 1

        # print('/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[0] + '_' + vid + '.npy')
        if isfile('/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[
            0] + '_' + vid + '.npy'):
            heatmap_batch[0, t, :, :, 0] = np.load(
                '/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[
                    0] + '_' + vid + '.npy')

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

            y_ori[0,t,:,:,0] = y_tmp

            if crop_margin > 0:
                y_tmp = y_tmp[crop_margin:-crop_margin, :]

            y_batch[0, t, :, :, 0] = cv2.resize(y_tmp, (28, 28))

    res_seq = mymodel.predict([heatmap_batch], batch_size=1)

    res_seq = res_seq[0][:, :, :, 0]

    # for ttt in range(time_step):
    #     res = res_seq[ttt, :, :]
    #     # plt.figure()
    #     # plt.imshow(res)
    #     # plt.show()
    #     y_res = np.zeros(shape=(320, 480))
    #     tt = cv2.resize(res, (480, (320 - 2 * int(crop_batch[0,ttt]))))
    #     y_res[int(crop_batch[0,ttt]):(320 - int(crop_batch[0,ttt])), :] = tt
    #     # plt.imshow(y_res)
    #     [tp_t, fp_t, ifframe, true_pos, detected_bbx, num_ca] = eval_bbx(y_res,list_now)
    #     dis = getDistance(y_res, y_ori[0,ttt,:,:,0])
    #
    #     ifframe_s = ifframe_s + ifframe
    #     true_pos_s = true_pos_s + true_pos
    #     detected_bbx_s = detected_bbx_s + detected_bbx
    #     num_ca_s = num_ca_s + num_ca
    #     distance = distance + dis
    #     tp = tp + tp_t
    #     fp = fp + fp_t
    #     if num_ca > 0:
    #         pos = pos + 1
    #     else:
    #         neg = neg + 1

    for tt in range(0,time_step):
        f_name = './Result/' + str(test_id) + '_' + str(index_start)
        if isdir(f_name) == False:
            os.mkdir(f_name)
        res=res_seq[tt,:,:]

        # plt.figure('test result for image {}'.format(list_now[0].split('/')[-1]))
        #
        # # plt.subplot(1, 4, 1)
        # plt.title('original image')

        # cv2.imshow('single img',x_batch[0, tt])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        # plt.figure('1')
        # plt.imshow(np.asarray(x_batch[0, tt,:,:],dtype='uint8'))
        # plt.title('x_batch')
        #
        # plt.show()

        # cv2.imshow('x_batch',np.asarray(x_batch[0, tt,:,:],dtype='uint8'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f_name+'/'+str(tt)+'.png', x_batch[0, tt, :, :, :])
        x_batch_ori=x_batch
        tmp=cv2.resize(2550*res_seq[tt, :,:], (224, 224))
        # print(np.min(tmp))
        # print(np.max(tmp))

        lamb = 0.7
        np.save(f_name+'/LSTM_res_'+str(tt),cv2.resize(res_seq[tt, :,:], (224, 224)))
        np.save(f_name + '/SF_res_'+str(tt), cv2.resize(heatmap_batch[0,tt, :, :,0], (224, 224)))
        prop = np.asarray(cv2.resize(prop_batch[0,tt, :, :,0], (224, 224), interpolation=cv2.INTER_NEAREST))
        prop[prop>=0.5] = 1
        prop[prop <= 0.5] = 0
        np.save(f_name + '/RP_hm_'+str(tt), prop)
        np.save(f_name + '/Gaze_hm_'+str(tt), cv2.resize(gaze_heatmap_batch[0, tt, :, :, 0], (224, 224)))


        #
        x_batch[0, tt, :, :, 0] = x_batch[0, tt, :, :, 0] *lamb + cv2.resize(2550*res_seq[tt,:,:], (224, 224))*(1-lamb)
        x_batch[0, tt, :, :, 1] = x_batch[0, tt, :, :, 1] *lamb+ cv2.resize(2550*res_seq[tt, :,:], (224, 224))*(1-lamb)
        x_batch[0, tt, :, :, 2] = x_batch[0, tt, :, :, 2] *lamb+ cv2.resize(2550*res_seq[tt, :,:], (224, 224))*(1-lamb)

        #cv2.imshow('x_batch',np.asarray(x_batch[0, tt],dtype='uint8'))
        # cv2.imshow('img',cv2.resize(255*res_seq[tt, :,:], (224, 224)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # plt.figure('1')
        # tmp=cv2.resize(255*res_seq[tt, :,:], (224, 224))
        # tmp3=np.zeros((224,224,3))
        # tmp3[:,:,0]=tmp
        # tmp3[:, :, 1] = tmp
        # tmp3[:, :, 2] = tmp
        # plt.imshow(np.asarray(tmp3,dtype='uint8'))
        # plt.title('x_batch')
        #
        # plt.show()


        # plt.imshow(x_batch[0, tt])  # +y_batch[j, :, :, :]/20)
        # plt.axis('off')

        # cv2.imshow('comb',np.asarray(x_batch[0, tt],dtype='uint8'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # fig, ax = plt.subplots()
        # data = np.asarray(cv2.resize(2550 * res_seq[tt, :, :], (224, 224)), dtype='uint8')
        #
        # cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot)#cm.coolwarm)#cm.viridis)#cm.afmhot)
        # ax.set_title('Gaussian noise with horizontal colorbar')
        #
        # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

        # plt.imshow(fig)
        #
        # plt.show()
        #
        # plt.figure('test')
        # plt.subplot(1,2,1)
        # plt.imshow(cax._A)
        #
        # plt.subplot(1,2,2)
        # plt.imshow(data)
        #
        # plt.show()
        #
        # tmp=cax._A

        plt.figure('demo')

        plt.subplot(1,5,1)
        plt.imshow(np.asarray(x_batch[0, tt,:,:,0],dtype='uint8'))
        plt.title('comb')

        plt.subplot(1,5,2)
        plt.imshow(np.asarray(cv2.resize(y_batch[0,tt], (224, 224)),dtype='uint8'))
        plt.title('gt')

        plt.subplot(1,5,3)
        plt.imshow(cv2.resize(res_seq[tt, :,:], (224, 224)))
        plt.title('res')

        plt.subplot(1, 5, 4)
        plt.imshow(prop)
        plt.title('res')

        plt.subplot(1, 5, 5)
        plt.imshow(cv2.resize(gaze_heatmap_batch[0,tt,:,:,0], (224, 224)))
        plt.title('res')

        plt.savefig(f_name+'/comb'+str(tt)+ '.jpg')

        if(tt%2 == 0):
            plt.show()

        print('t {}'.format(tt))

        """Produce custom labelling for a colorbar.

        Contributed by Scott Sinclair
        """



        # # Make plot with vertical (default) colorbar
        # fig, ax = plt.subplots()
        #
        # tmp=res_seq[tt, :,:]
        # data = np.asarray(cv2.resize(2550*res_seq[tt, :,:], (224, 224)),dtype='uint8')
        #
        # cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
        # ax.set_title('Gaussian noise with vertical colorbar')
        #
        # # Add colorbar, make sure to specify tick locations to match desired ticklabels
        # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

        # Make plot with horizontal colorbar






        # plt.subplot(1, 4, 2)
        # plt.figure()
        # plt.title('gazeheatmap')
        # plt.imshow(cv2.resize(heatmap_batch[0, tt], (224, 224)))
        # plt.axis('off')
        #
        # # plt.subplot(1, 4, 3)
        # plt.figure()
        # plt.title('gt coatt')
        # plt.imshow(cv2.resize(y_batch[0, tt], (224, 224)))
        # plt.axis('off')

        # # plt.subplot(1, 4, 4)
        # plt.figure()
        # plt.title('res coatt')
        # plt.imshow(cv2.resize(res_seq[tt, :,:], (224, 224)))
        # plt.axis('off')
        #
        # plt.show()
# plt.figure()
# # print(sum(num_ca_s))
# ifframe_s = ifframe_s / float(nb_test_samples)
# recall = true_pos_s / float(num_ca_s)
# dis_avg = distance / float(num_ca_s)
# precision = true_pos_s / detected_bbx_s
# filename = '/home/lfan/Dropbox/runCoAtt/new_experiment/' + version +'precision'
# # np.save(filename,precision)
# filename = '/home/lfan/Dropbox/runCoAtt/new_experiment/' + version + 'recall'
# # np.save(filename, recall)
# x_axis = np.linspace(1 / RANGE, 1, RANGE)
# x_axis = np.reshape(x_axis, (1, RANGE))
#
# dis_x_axis = np.linspace(1 / DIS_RANGE, 1, DIS_RANGE)
# dis_x_axis = np.reshape(dis_x_axis, (1, DIS_RANGE))
#
# tpr = tp / float(pos)
# fpr = fp / float(neg)
#
# plt.subplot(1, 4, 1)
# plt.title('frame acc')
# plt.plot(x_axis, ifframe_s, 'ro')
# plt.subplot(1, 4, 2)
# plt.title('prec_recall')
# plt.plot(recall, precision, 'ro')
# plt.subplot(1, 4, 3)
# plt.title('distance')
# plt.plot(dis_x_axis, dis_avg, 'ro')
# plt.subplot(1, 4, 4)
# plt.title('roc')
# plt.plot(tpr, fpr, 'ro')
# plt.show()
#
# print(np.max(ifframe_s))
# # print(np.argmin(dis_avg))
# print(distance/float(num_ca_s))
# TODO 

