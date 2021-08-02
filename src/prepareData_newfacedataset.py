import os
from os import listdir
from os.path import isfile, join, isdir
import cv2

# ### validation
# file_to_read_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/wider_face_split/wider_face_val_bbx_gt.txt'
#
# with open(file_to_read_path,'r') as file_to_read:
#   with open('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/validate.txt','w') as train_list:
#
#     while True:
#         line=file_to_read.readline()
#         line=line[:-1]
#         if len(line)==0:
#             break
#         img_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_val/images/'+line
#         train_list.write(img_path+'\n')
#         img=cv2.imread(img_path)
#         img_height, img_width, ch=img.shape
#         dh=1./img_height
#         dw=1./img_width
#
#         txt_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_val/labels/'+line[:-4]+'.txt'
#         subfolder=line.split('/')[0]
#         if not isdir(join('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_val/labels/',subfolder)):
#             os.mkdir(join('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_val/labels/',subfolder))
#         line=file_to_read.readline()
#         face_num=int(line)
#         with open(txt_path,'w') as file_to_write:
#
#           for face_id in range(face_num):
#             line=file_to_read.readline()
#             line_list=line.split()
#
#             blur=int(line_list[4])
#             invalid=int(line_list[7])
#             occlusion=int(line_list[8])
#             if blur<2 and invalid==0 and occlusion<2:
#                 x = float(line_list[0])
#                 y = float(line_list[1])
#                 w = float(line_list[2])
#                 h = float(line_list[3])
#                 x_tmp=(x+w/2.0-1)*dw
#                 y_tmp=(y+h/2.0-1)*dh
#                 w_tmp=w*dw
#                 h_tmp=h*dh
#                 file_to_write.write('0' + " " + str(x_tmp) + " " + str(y_tmp) + " " + str(w_tmp) + " " + str(h_tmp) + '\n')


### train
file_to_read_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/wider_face_split/wider_face_train_bbx_gt.txt'

with open(file_to_read_path,'r') as file_to_read:
  with open('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/train.txt','w') as train_list:

    while True:
        line=file_to_read.readline()
        line=line[:-1]
        if len(line)==0:
            break
        img_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_train/images/'+line
        train_list.write(img_path+'\n')
        img=cv2.imread(img_path)
        img_height, img_width, ch=img.shape
        dh=1./img_height
        dw=1./img_width

        txt_path='/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_train/labels/'+line[:-4]+'.txt'
        subfolder=line.split('/')[0]
        if not isdir(join('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_train/labels/',subfolder)):
            os.mkdir(join('/home/lfan/Dropbox/JointAttention/Data/face_detect_dataset/WIDER_train/labels/',subfolder))
        line=file_to_read.readline()
        face_num=int(line)
        with open(txt_path,'w') as file_to_write:

          for face_id in range(face_num):
            line=file_to_read.readline()
            line_list=line.split()

            blur=int(line_list[4])
            invalid=int(line_list[7])
            occlusion=int(line_list[8])
            if blur<2 and invalid==0 and occlusion<2:
                x = float(line_list[0])
                y = float(line_list[1])
                w = float(line_list[2])
                h = float(line_list[3])
                x_tmp=(x+w/2.0-1)*dw
                y_tmp=(y+h/2.0-1)*dh
                w_tmp=w*dw
                h_tmp=h*dh
                file_to_write.write('0' + " " + str(x_tmp) + " " + str(y_tmp) + " " + str(w_tmp) + " " + str(h_tmp) + '\n')


