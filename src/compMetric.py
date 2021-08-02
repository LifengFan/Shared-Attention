from keras.utils.data_utils import get_file
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
from random import shuffle
import math
import matplotlib.pyplot as plt

nb_test_pos = 2500
nb_test_neg = 2000
nb_test_samples = 4500

def getGazeHeatmap(face_loc, gaze_dir):
    sigma = 0.3
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

## prepare testing data
file_name_pos = '/home/lfan/Dropbox/JointAttention/Data/test_coatt_summary_pos.txt'
file_name_neg = '/home/lfan/Dropbox/JointAttention/Data/test_coatt_summary_neg.txt'
prop_path = '/home/lfan/Dropbox/JointAttention/bbx_prop_clear/'

with open(file_name_pos, 'r') as data_to_read:
    data_lines_pos = data_to_read.readlines()
with open(file_name_neg, 'r') as data_to_read:
    data_lines_neg = data_to_read.readlines()

shuffle(data_lines_pos)
shuffle(data_lines_neg)

data_lines_pos = data_lines_pos[0:nb_test_pos]
data_lines_neg = data_lines_neg[0:nb_test_neg]

#data_lines_pos[0:0] = data_lines_neg
data_lines = data_lines_pos
shuffle(data_lines)

TPR_ALL=[]
FPR_ALL=[]



for thres in np.linspace(0, 0.999, num=5):

    # recall_all = 0
    # precision_all = 0
    # CNT = 0

    tp_area = 0
    gt_area = 0
    select_area = 0

    for test_id in range(len(data_lines)):

        line_now = data_lines[test_id]
        list_now = line_now.split()
        vid = list_now[0].split('_')[-1][:-4]

        # get ground truth result
        gt_ca_heatmap = np.zeros(shape=(320, 480))

        if isfile('/home/lfan/Dropbox/JointAttention/Data/coatt_per_frame/' + vid + '/' + str(
                int(list_now[0].split('/')[-1].split('_')[0])) + '.txt'):
            with open('/home/lfan/Dropbox/JointAttention/Data/coatt_per_frame/' + vid + '/' + str(
                    int(list_now[0].split('/')[-1].split('_')[0])) + '.txt', 'r') as gt_to_read:
                gt_lines = gt_to_read.readlines()

            for s in range(len(gt_lines)):
                gt_list = gt_lines[s].split()
                ca_xmin = int(gt_list[0])
                ca_ymin = int(gt_list[1])
                ca_xmax = int(gt_list[2])
                ca_ymax = int(gt_list[3])

                gt_ca_heatmap[ca_ymin:ca_ymax, ca_xmin:ca_xmax] = 1

        gt_ca_heatmap = cv2.resize(gt_ca_heatmap, (28, 28))


        prop_heatmap = np.zeros(shape=(28, 28))
        if isfile(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt'):
            with open(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt', 'r') as f2:
                while True:
                    line = f2.readline()
                    if len(line) == 0:
                        break
                    list = line.split()
                    xmin = float(list[0])
                    ymin = float(list[1])
                    xmax = float(list[2])
                    ymax = float(list[3])

                    xmin = int(xmin * (28. / 480))
                    ymin = int(ymin * (28. / (320 )))
                    xmax = int(xmax * (28. / 480))
                    ymax = int(ymax * (28. / (320 )))

                    xmin = min(27, xmin)
                    ymin = min(27, ymin)
                    xmax = max(xmin + 1, xmax)
                    ymax = max(ymin + 1, ymax)

                    prop_heatmap[ymin:ymax, xmin:xmax] =  float(list[4])


        if isfile('/home/lfan/Dropbox/JointAttention/tested_face_direction/test/' + list_now[0].split('/')[-1][:-4] + '.txt'):
            with open('/home/lfan/Dropbox/JointAttention/tested_face_direction/test/' + list_now[0].split('/')[-1][:-4] + '.txt','r') as gaze_to_read:
                gaze_lines = gaze_to_read.readlines()

            for gaze_id in range(len(gaze_lines)):
                gaze_list = gaze_lines[gaze_id].split()

                g_xmin = int(float(gaze_list[3]) * (28.0 / 480))
                g_ymin = int(float(gaze_list[4]) * (28.0 / 320))
                g_xmax = int(float(gaze_list[5]) * (28.0 / 480))
                g_ymax = int(float(gaze_list[6]) * (28.0 / 320))

                prop_heatmap[g_ymin:g_ymax, g_xmin:g_xmax] = 1



        if isfile('/home/lfan/Dropbox/JointAttention/tested_face_direction/test/' + list_now[0].split('/')[-1][:-4] + '.txt'):
            with open('/home/lfan/Dropbox/JointAttention/tested_face_direction/test/' + list_now[0].split('/')[-1][:-4] + '.txt','r') as gaze_to_read:
                gaze_lines = gaze_to_read.readlines()

            gazeheatmap = np.zeros(shape=(28, 28))

            for gaze_id in range(len(gaze_lines)):
                gaze_list = gaze_lines[gaze_id].split()

                g_xmin = float(gaze_list[3]) * (24.0 / 480)
                g_ymin = float(gaze_list[4]) * (16.0 / 320)
                g_xmax = float(gaze_list[5]) * (24.0 / 480)
                g_ymax = float(gaze_list[6]) * (16.0 / 320)

                g_x = (g_xmin + g_xmax) / 2
                g_y = (g_ymin + g_ymax) / 2

                dir_x = float(gaze_list[7])
                dir_y = float(gaze_list[8])

                gazehp = getGazeHeatmap([g_x, g_y], [dir_x, dir_y])
                gazehp = cv2.resize(gazehp, (28, 28))
                gazeheatmap += gazehp
                # ghp_cnt+=1

            max_val = np.max(gazeheatmap)

            if max_val!=0:
                gazeheatmap = gazeheatmap / max_val

            # print(gazeheatmap)

            # plt.figure(list_now[0])
            #
            # plt.subplot(1, 2, 1)
            # plt.title('gt ca heatmap')
            # plt.imshow(gt_ca_heatmap)
            #
            # plt.subplot(1, 2, 2)
            # plt.title('gaze heatmap')
            # plt.imshow(cv2.resize(gazeheatmap, (480, 320)))
            #
            # plt.axis('off')
            #
            # plt.show()

            gazeheatmap=np.multiply(gazeheatmap,prop_heatmap)

        gaze_selected = np.zeros(shape=(28, 28))
        gaze_selected[gazeheatmap > thres] = 1

        tp = np.sum(np.multiply(gaze_selected, gt_ca_heatmap))

        tp_area+=tp
        gt_area+=np.sum(gt_ca_heatmap)
        select_area+=np.sum(gaze_selected)

        # if np.sum(gt_ca_heatmap)==0:
        #     recall=0
        # else:
        #     recall=tp / np.sum(gt_ca_heatmap)
        #
        # if np.sum(gaze_selected)==0:
        #     precision=0
        # else:
        #     precision=tp / np.sum(gaze_selected)


        # recall_all += recall
        # precision_all += precision
        #CNT += 1

    final_recall = tp_area/gt_area
    final_precision = tp_area/select_area

    print('thres: {}   '.format(thres))
    print('Average recall: {}  '.format(final_recall))
    print('Average precision: {}  '.format(final_precision))

    TPR_ALL.append(final_recall)
    FPR_ALL.append(1-final_precision)


plt.figure(figsize=(4, 4), dpi=80)
plt.xlabel("FPR", fontsize=14)
plt.ylabel("TPR", fontsize=14)
plt.title("ROC Curve", fontsize=14)

plt.plot(FPR_ALL, TPR_ALL, color='red', linewidth=2, label='gazeheatmap')

x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='green', linewidth=2, label='random')

plt.xlim(0.0, 1.2)
plt.ylim(0.0, 1.2)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.savefig('gazeheatmap_tested_roc_curve5.jpg')


