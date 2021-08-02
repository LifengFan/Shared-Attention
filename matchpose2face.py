import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2

pose_path='/home/lfan/Dropbox/JointAttention/Data/2dpose/'
pose_path_order='/home/lfan/Dropbox/JointAttention/Data/2dpose_order/'
pose_sf=sorted(listdir(pose_path))
face_path='/home/lfan/Dropbox/JointAttention/faces/test/'
face_sf=sorted(listdir(face_path))

for i in range(len(face_sf)):
    sf=face_sf[i]
    face_files=sorted(listdir(face_path+sf))
    for j in range(len(face_files)):
        face_file=face_files[j]
        with open(face_path+sf+'/'+face_file,'r') as face_to_read:
            face_lines=face_to_read.readlines()

            if  isfile(pose_path+sf+'/'+face_file.split('_')[0]+'.txt'):

                with open(pose_path+sf+'/'+face_file.split('_')[0]+'.txt','r') as pose_to_read:
                    pose_lines=pose_to_read.readlines()

                    nrow = len(face_lines)
                    ncol = len(pose_lines)

                    if nrow!=0 and ncol!=0:
                        face_set = []
                        pose_set = []

                        for k in range(len(face_lines)):
                            face_line = face_lines[k]
                            face_list = face_line.split()

                            face_xmin = float(face_list[3])
                            face_ymin = float(face_list[4])
                            face_xmax = float(face_list[5])
                            face_ymax = float(face_list[6])

                            face_x = (face_xmin + face_xmax) / 2
                            face_y = (face_ymin + face_ymax) / 2

                            face_set.append([face_x, face_y])

                        for m in range(len(pose_lines)):
                            pose_line = pose_lines[m]
                            # compute_dist(line,pose_line)
                            pose_list = pose_line.split()

                            pose_x = float(pose_list[27])
                            pose_y = float(pose_list[28])

                            pose_set.append([pose_x, pose_y])

                        distmat = 10000 * np.ones((nrow, ncol))

                        for k in range(nrow):
                            for m in range(ncol):
                                distmat[k, m] = np.sqrt(
                                    np.sum((np.asarray(face_set[k]) - np.asarray(pose_set[m])) ** 2))

                        map_dict = {}
                        max_val = distmat.max() + 1000
                        while True:
                            if distmat.min() == max_val:
                                break
                            ind = distmat.argmin()
                            ind_col = (ind + 1) % ncol
                            if ind_col == 0:
                                ind_col = ncol - 1
                                ind_row = (ind + 1) / ncol - 1
                            else:
                                ind_col = ind_col - 1
                                ind_row = (ind + 1) / ncol

                            map_dict[ind_row] = ind_col

                            distmat[ind_row, :] = max_val
                            distmat[:, ind_col] = max_val

                        if not isdir(pose_path_order + sf + '/'):
                            os.mkdir(pose_path_order + sf + '/')
                        with open(pose_path_order + sf + '/' + face_file.split('_')[0] + '.txt', 'w') as pose_to_write:
                            for faceid in range(len(face_lines)):
                                if map_dict.has_key(faceid):
                                    poseid = map_dict[faceid]
                                    pose_to_write.write(pose_lines[poseid])
                                else:
                                    pose_to_write.write('0\t' * 48 + '\n')

                    elif nrow!=0 and ncol==0:
                        if not isdir(pose_path_order + sf + '/'):
                            os.mkdir(pose_path_order + sf + '/')
                        with open(pose_path_order + sf + '/' + face_file.split('_')[0] + '.txt', 'w') as pose_to_write:
                            for faceid in range(len(face_lines)):
                                pose_to_write.write('0\t' * 48 + '\n')




































