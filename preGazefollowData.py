import os
import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join, isdir
import multiprocessing
import sys


dataPath='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/'
trainannotpath=dataPath+'trainAnnot_new.txt'
testannotpath=dataPath+'testAnnot_new.txt'
savepath_train='/home/lfan/Dropbox/runCoAtt/vgg16/8class_gazefollow/train/'
savepath_test='/home/lfan/Dropbox/runCoAtt/vgg16/8class_gazefollow/test/'

nb_bin=8
angle_per_bin=(2*math.pi)/nb_bin
faceCNT_train=np.zeros(nb_bin)
faceCNT_test=np.zeros(nb_bin)

for direction in range(nb_bin):
    imagesavepath = savepath_train + str(direction+1) + '/'
    if not os.path.exists(imagesavepath):
      os.mkdir(imagesavepath)

for direction in range(nb_bin):
    imagesavepath = savepath_test + str(direction + 1) + '/'
    if not os.path.exists(imagesavepath):
        os.mkdir(imagesavepath)

with open(trainannotpath,'r') as file_to_read:
        lines_train = file_to_read.readlines()
with open(testannotpath, 'r') as file_to_read:
        lines_test = file_to_read.readlines()

def preTrainData(id):
        line=lines_train[id]
        line=line[:-1]
        line_list=line.split(' ')
        name=line_list[0]

        bbox_x=int(line_list[1])
        bbox_y=int(line_list[2])
        bbox_w=int(line_list[3])
        bbox_h=int(line_list[4])

        eyes_x=float(line_list[5])
        eyes_y=float(line_list[6])

        gaze_x=float(line_list[7])
        gaze_y=float(line_list[8])

        direction=float(line_list[9])

        direction=int(math.ceil(direction/angle_per_bin))
        if direction==0:
            direction=nb_bin
        if direction > nb_bin:
            direction = nb_bin
        img=cv2.imread(dataPath+name)
        h,w,ch=img.shape

        bbox_xmin=bbox_x
        bbox_ymin=bbox_y
        bbox_xmax=bbox_x+bbox_w
        bbox_ymax=bbox_y+bbox_h

        if bbox_xmin<0:
            bbox_xmin=0
        elif bbox_xmax>=w:
            bbox_xmax=w
        if bbox_ymin<0:
            bbox_ymin=0
        elif bbox_ymax>=h:
            bbox_ymax=h

        imgpatch=img[bbox_ymin:bbox_ymax,bbox_xmin:bbox_xmax,:]
        faceCNT_train[direction - 1] += 1
        cv2.imwrite(savepath_train + str(direction) + '/'+str(int(faceCNT_train[direction - 1])) + '.jpg',imgpatch)


def preTestData(id):
    line = lines_test[id]
    line = line[:-1]
    line_list = line.split(' ')
    name = line_list[0]

    bbox_x = int(line_list[1])
    bbox_y = int(line_list[2])
    bbox_w = int(line_list[3])
    bbox_h = int(line_list[4])

    eyes_x = float(line_list[5])
    eyes_y = float(line_list[6])

    gaze_x = float(line_list[7])
    gaze_y = float(line_list[8])

    direction = float(line_list[9])

    direction = int(math.ceil(direction / angle_per_bin))
    if direction == 0:
        direction = nb_bin
    if direction>nb_bin:
        direction=nb_bin
    img = cv2.imread(dataPath + name)
    h, w, ch = img.shape

    bbox_xmin = bbox_x
    bbox_ymin = bbox_y
    bbox_xmax = bbox_x + bbox_w
    bbox_ymax = bbox_y + bbox_h

    if bbox_xmin < 0:
        bbox_xmin = 0
    elif bbox_xmax >= w:
        bbox_xmax = w
    if bbox_ymin < 0:
        bbox_ymin = 0
    elif bbox_ymax >= h:
        bbox_ymax = h

    imgpatch = img[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]
    faceCNT_test[direction - 1] += 1
    cv2.imwrite(savepath_test + str(direction) + '/' + str(int(faceCNT_test[direction - 1])) + '.jpg', imgpatch)

#
# cores=multiprocessing.cpu_count()
# pool=multiprocessing.Pool(processes=cores)
# cnt = 0
#
# for _ in pool.imap(preTrainData,range(len(lines_train))):
#     sys.stdout.write('done %d/%d\r' % (cnt, len(lines_train)))
#     cnt += 1
#
# cnt = 0
# for _ in pool.imap(preTestData,range(len(lines_test))):
#     sys.stdout.write('done %d/%d\r' % (cnt, len(lines_test)))
#     cnt += 1

for id in range(len(lines_test)):
    preTestData(id)
    print(id)


#preTrainData(0)

