from keras.utils.data_utils import get_file
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
from random import shuffle
import math
import matplotlib.pyplot as plt
from keras.utils.data_utils import get_file
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
from random import shuffle
import math
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda
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
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='prediction')(x)

base_model = Model(inputs=img_input, outputs=output)
base_model.summary()

# base_model.load_weights(model_weights_path)#, by_name=True
base_model.load_weights('/home/lfan/Dropbox/runCoAtt/new_experiment/' + 'v1' + '/checkpoint.weights.21-0.04.hdf5')


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

for thres in np.linspace(0, 0.999, num=50):

    tp_area = 0
    gt_area = 0
    select_area = 0

    for test_id in range(len(data_lines)):

        line_now = data_lines[test_id]
        list_now = line_now.split()
        vid = list_now[0].split('_')[-1][:-4]


        img = cv2.imread(list_now[0])

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

        img = img.astype('float32')
        img = cv2.resize(img, (224, 224))
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

        x_batch = np.zeros(shape=(1, 224, 224, 3))
        x_batch[0, :, :, :] = img
        res = base_model.predict(x_batch, batch_size=1)
        tested_heatmap = res[0][:, :, 0]



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

        tested_selected = np.zeros(shape=(28, 28))
        tested_selected[tested_heatmap > thres] = 1

        tp = np.sum(np.multiply(tested_selected, gt_ca_heatmap))

        tp_area+=tp
        gt_area+=np.sum(gt_ca_heatmap)
        select_area+=np.sum(tested_selected)



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
plt.savefig('gazeheatmap_tested_roc_curve_rawimagebaseline.jpg')



