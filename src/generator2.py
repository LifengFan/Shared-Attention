from os.path import isfile, join, isdir
import numpy as np
import cv2
from gazemap import getGazeHeatmap
from random import shuffle
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt
nb_train_pos = 7000
nb_train_neg = 2000
nb_train_samples = 9000

nb_validate_pos = 4500
nb_validate_neg = 1200
nb_validate_samples = 5700

nb_test_pos = 2000
nb_test_neg = 1000
nb_test_samples = 4000

batch_size = 25
epochs = 40

images_train_path = '/home/yixin/Projects/CVPR2018/cm/frames/train/'
images_validate_path = '/home/yixin/Projects/CVPR2018/cm/frames/validate/'
images_test_path = '/home/yixin/Projects/CVPR2018/cm/frames/test/'

prop_path = '/media/yixin/Elements/bbx_proposal/'
# train: mode=1
# validate: mode=2
def my_generator(mode):
    if mode == 1:
        file_name_pos = 'train_coatt_summary_pos.txt'
        file_name_neg = 'train_coatt_summary_neg.txt'
        gaze_dir_prefix = '/home/yixin/Dropbox/JointAttention/tested_face_direction/train/'

    elif mode == 2:
        file_name_pos = 'validate_coatt_summary_pos.txt'
        file_name_neg = 'validate_coatt_summary_neg.txt'
        gaze_dir_prefix = '/home/yixin/Dropbox/JointAttention/tested_face_direction//validate/'

    elif mode == 3:
        file_name_pos = 'test_coatt_summary_pos.txt'
        file_name_neg = 'test_coatt_summary_neg.txt'
        gaze_dir_prefix = '/home/yixin/Dropbox/JointAttention/tested_face_direction/test/'

    with open(file_name_pos, 'r') as data_to_read:
        data_lines_pos = data_to_read.readlines()
    with open(file_name_neg, 'r') as data_to_read:
        data_lines_neg = data_to_read.readlines()

    shuffle(data_lines_pos)
    shuffle(data_lines_neg)

    if mode == 1:
        data_lines_pos = data_lines_pos[0:nb_train_pos]
        data_lines_neg = data_lines_neg[0:nb_train_neg]

    elif mode == 2:
        data_lines_pos = data_lines_pos[0:nb_validate_pos]
        data_lines_neg = data_lines_neg[0:nb_validate_neg]
    elif mode == 3:
        data_lines_pos = data_lines_pos[0:nb_test_pos]
        data_lines_neg = data_lines_neg[0:nb_test_neg]

    data_lines_pos[0:0] = data_lines_neg
    data_lines = data_lines_pos

    shuffle(data_lines)

    cur_batch_index = 0

    while True:

        x_batch = np.zeros(shape=(batch_size, 224, 224, 3))
        # gaze_heatmap_batch = np.zeros(shape=(batch_size, 224, 224, 1))
        # prop_batch = np.zeros(shape=(batch_size, 224, 224, 1))
        gaze_heatmap_batch = np.zeros(shape=(batch_size, 28, 28, 1))
        prop_batch = np.zeros(shape=(batch_size, 28, 28, 1))

        if mode == 1 or mode == 2:
            y_batch = np.zeros(shape=(batch_size, 28, 28, 1))


        start_id = cur_batch_index * batch_size
        end_id = start_id + batch_size
        files_batch_now = data_lines[start_id:end_id]

        for j in range(batch_size):

            list_now = files_batch_now[j].split()
            vid = list_now[0].split('_')[-1][:-4]

            ### image part
            if  list_now[0].split('/')[-3] == 'train':
                imgfilename = images_train_path+list_now[0].split('/')[-1][0:5]+'_'+list_now[0].split('/')[-1][6:-4]+'.jpg'
            if  list_now[0].split('/')[-3] == 'validate':
                imgfilename = images_validate_path + list_now[0].split('/')[-1][0:5] + '_' + list_now[0].split('/')[-1][6:-4] + '.jpg'
            if isfile(imgfilename) == False:
                print(imgfilename)
            img = cv2.imread(imgfilename)
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
            # img = img.astype('float32')
            img = cv2.resize(img, (224, 224))

            x_batch[j, :, :, :] = img

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
                # gaze_heatmap_batch[j, :, :, 0] = cv2.resize(gazeheatmap, (224, 224))
                gaze_heatmap_batch[j, :, :, 0] = cv2.resize(gazeheatmap, (28, 28))

            ### face bbx part
            face_tmp = np.zeros(shape=(320, 480))
            if isfile(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt'):
                with open(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt', 'r') as gaze_to_read:
                    gaze_lines = gaze_to_read.readlines()



                for gaze_id in range(len(gaze_lines)):
                    gaze_list = gaze_lines[gaze_id].split()

                    cv2.rectangle(img_orig, (int(gaze_list[3]), int(gaze_list[4])),
                                  (int(gaze_list[5]), int(gaze_list[6])), (0, 255, 255))

                    g_xmin = int(float(gaze_list[3]))
                    g_ymin = int(float(gaze_list[4]))
                    g_xmax = int(float(gaze_list[5]))
                    g_ymax = int(float(gaze_list[6]))

                    face_tmp[g_ymin:g_ymax, g_xmin:g_xmax] = 1

                # if crop_margin > 0:
                #     face_tmp = face_tmp[crop_margin:-crop_margin, :]

                # prop_batch[j, :, :, 0] = cv2.resize(face_tmp, (224, 224))

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

            if crop_margin > 0:
                face_tmp = face_tmp[crop_margin:-crop_margin, :]
            # prop_batch[j, :, :, 0] = cv2.resize(face_tmp, (224, 224))
            prop_batch[j, :, :, 0] = cv2.resize(face_tmp, (28, 28))

            ### add gaussian filter to prop heatmap

            # prop_batch[j, :, :, 0] = gaussian(prop_batch[j, :, :, 0], 2)

            ### gt coatt part

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

                    y_batch[j, :, :, 0] = cv2.resize(y_tmp, (28, 28))
                    # ### add gaussian filter to groundtruth
                    # y_batch[j, :, :, 0] = gaussian(y_batch[j, :, :, 0], 1)



            # plt.figure(num=str(j))
            #
            # plt.subplot(2, 3, 1)
            # plt.title('original image')
            # plt.imshow(img) #+y_batch[j, :, :, :]/20)
            #
            # plt.subplot(2, 3, 2)
            # plt.title('gazeheatmap')
            # plt.imshow(cv2.resize(gaze_heatmap_batch[j, :, :],(224,224)))
            #
            # plt.subplot(2, 3, 3)
            # plt.title('propheatmap')
            # plt.imshow(cv2.resize(prop_batch[j, :, :],(224,224)))
            #
            # # plt.subplot(2, 3, 4)
            # # plt.title('facepropheatmap')
            # # plt.imshow(cv2.resize(face_prop_batch[j, :, :], (224, 224)))
            #
            # plt.subplot(2, 3, 5)
            # plt.title('gt coatt')
            # plt.imshow(cv2.resize(y_batch[j, :, :],(224,224)))
            #
            # g = lambda x: np.multiply(x[0], x[1])
            # plt.subplot(2, 3, 6)
            # plt.title('multiply')
            # plt.imshow(cv2.resize(g([gaze_heatmap_batch[j, :, :],prop_batch[j, :, :]]),(224,224)))
            #
            # plt.axis('off')
            # plt.show()

        # yield [x_batch, gaze_heatmap_batch, prop_batch], y_batch
        if mode == 1 or mode == 2:
            yield [prop_batch], y_batch
            # yield [gaze_heatmap_batch, prop_batch], y_batch
            # yield [x_batch, gaze_heatmap_batch, prop_batch], y_batch
        elif mode == 3:
            yield [gaze_heatmap_batch, prop_batch]

        cur_batch_index = cur_batch_index + 1
        if cur_batch_index >= (len(data_lines) // batch_size):
            cur_batch_index = 0
            shuffle(data_lines)