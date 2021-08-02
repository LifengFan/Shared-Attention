import numpy as np
from random import shuffle
from os.path import isfile, join, isdir
import cv2
import matplotlib.pyplot as plt
time_step = 10

def my_generator(mode,batch_size):
    if mode == 1:
        file_name = '/home/lfan/Dropbox/JointAttention/Data/train_coatt_summary_contin.txt'
        index_set = np.load('/home/lfan/Dropbox/JointAttention/Data/train_index.npy')
        gaze_dir_prefix = '/home/lfan/Dropbox/JointAttention/tested_face_direction/train/'

    elif mode == 2:
        file_name = '/home/lfan/Dropbox/JointAttention/Data/validate_coatt_summary_contin.txt'
        index_set = np.load('/home/lfan/Dropbox/JointAttention/Data/validate_index.npy')
        gaze_dir_prefix = '/home/lfan/Dropbox/JointAttention/tested_face_direction/validate/'

    elif mode==3:
        file_name = '/home/lfan/Dropbox/JointAttention/Data/test_coatt_summary_contin.txt'
        index_set = np.load('/home/lfan/Dropbox/JointAttention/Data/test_index.npy')
        gaze_dir_prefix = '/home/lfan/Dropbox/JointAttention/tested_face_direction/test/'


    with open(file_name, 'r') as data_to_read:
        data_lines = data_to_read.readlines()



    # if mode == 1:
    #     data_lines = data_lines[0:nb_train_samples]
    # elif mode == 2:
    #     data_lines = data_lines[0:nb_validate_samples]
    # elif mode==3:
    #     data_lines= data_lines[0:nb_test_samples]


    shuffle(index_set)

    cur_batch_index = 0
    while True:

        x_batch = np.zeros(shape=(batch_size, time_step,224, 224, 3))
        heatmap_batch = np.zeros(shape=(batch_size, time_step,28, 28, 1))
        if mode==1 or mode == 2 :
            y_batch = np.zeros(shape=(batch_size,time_step, 28, 28,1))

        start_id = cur_batch_index * batch_size

        for j in range(batch_size):

            index_start=index_set[start_id+j]

            for t in range(time_step):

                list_now = data_lines[index_start * time_step + t].split()
                # list_now = files_batch_now[j].split()
                vid = list_now[0].split('_')[-1][:-4]

                ### image part

                img = cv2.imread(list_now[0])
                img_orig = img

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

                x_batch[j, t,:, :, :] = img
                # print('/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[0] + '_' + vid + '.npy')
                if isfile('/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[0] + '_' + vid + '.npy'):
                        heatmap_batch[j, t,:, :, 0] = np.load('/home/lfan/Dropbox/runCoAtt/new_experiment/heatmap_m2_yxc/' + list_now[0].split('/')[-1].split('_')[0] + '_' + vid + '.npy')

                if mode == 1 or mode == 2:

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

                        if crop_margin > 0:
                            y_tmp = y_tmp[crop_margin:-crop_margin, :]

                        y_batch[j, t,:, :, 0] = cv2.resize(y_tmp, (28, 28))

                # plt.figure(num=str(t))
                #
                # plt.subplot(1, 3, 1)
                # plt.title('original image')
                # plt.imshow(img) #+y_batch[j, :, :, :]/20)
                # # plt.imshow(x_batch[j]) #+y_batch[j, :, :, :]/20)
                #
                # plt.subplot(1, 3, 2)
                # plt.title('heatmap')
                # plt.imshow(cv2.resize(heatmap_batch[j, t, :, :],(224,224)))
                #
                # plt.subplot(1, 3, 3)
                # plt.title('gt coatt')
                # plt.imshow(cv2.resize(y_batch[j, t, :, :],(224,224)))
                #
                # plt.axis('off')
                # plt.show()


        #yield [x_batch, gaze_heatmap_batch, prop_batch], y_batch
        if mode==1 or mode==2:
            yield [heatmap_batch], y_batch
        elif mode==3:
            yield [heatmap_batch]

        cur_batch_index = cur_batch_index + 1
        if cur_batch_index >= (len(index_set)// batch_size):
            cur_batch_index = 0
            shuffle(index_set)