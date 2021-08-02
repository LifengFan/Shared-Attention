import cv2
import numpy as np
import math

annot_train='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/trainAnnot_new.txt'
annot_test='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/testAnnot_new.txt'
datapath='/home/lfan/Dropbox/runCoAtt/vgg16/gazefollow_data/'

with open(annot_train,'r') as file_to_read:
    lines_train=file_to_read.readlines()
with open(annot_test,'r') as file_to_read:
    lines_test=file_to_read.readlines()


nb_train_samples=len(lines_train)
nb_test_samples=len(lines_test)

for i in range(len(lines_test)):
    line=lines_test[i]
    line = line[:-1]
    line_list = line.split(' ')

    name = line_list[0]
    bbox_x = int(line_list[1])
    bbox_y = int(line_list[2])
    bbox_w = int(line_list[3])
    bbox_h = int(line_list[4])
    eye_x = int(np.float32(line_list[5]))
    eye_y = int(np.float32(line_list[6]))

    # if bbox_w < bbox_h:
    #    facelen = bbox_w
    # else:
    #    facelen = bbox_h

    #face_x = int(math.ceil(eye_x - facelen * 0.4))
    #face_y = int(math.ceil(eye_y - facelen * 0.4))

    totop=eye_y-bbox_y
    face_h=int(2.5*totop)
    face_w=int(2.2*totop)
    face_x=int(eye_x-totop*1.5)
    face_y=int(eye_y-totop)

    # face_x=int(eye_x-bbox_w*0.3)
    # face_y=int(eye_y-bbox_h*0.2)
    # face_w=int(bbox_w*0.6)
    # face_h=int(bbox_h*0.4)

    # face_r=70
    #
    # face_x=eye_x-face_r
    # face_y=eye_y-face_r
    # face_w=2*face_r
    # face_h=2*face_r


    if face_x<bbox_x:
        face_x=bbox_x
    if face_y<bbox_y:
        face_y=bbox_y
    if (face_x+face_w)>(bbox_x+bbox_w):
        face_w=bbox_x+bbox_w-face_x
    if (face_y+face_h)>(bbox_y+bbox_h):
        face_h=bbox_y+bbox_h-face_y

    #face_w = int(facelen * 0.6)
    #face_h = int(facelen * 0.8)

    img = cv2.imread(datapath + name)
    cv2.rectangle(img,(bbox_x,bbox_y),(bbox_x+bbox_w,bbox_y+bbox_h),(0,255,0),3)
    cv2.rectangle(img,(eye_x-10,eye_y-10),(eye_x+10,eye_y+10),(255,0,0),3)
    cv2.rectangle(img,(face_x,face_y),(face_x+face_w,face_y+face_h),(0,0,255),3)
    cv2.imshow('fullimage',img)

    face_pro=img[face_x:(face_x+face_w),face_y:(face_y+face_h),:]
    #cv2.imshow('face',face_pro)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


