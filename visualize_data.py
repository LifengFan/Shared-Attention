import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2

data_path='/home/lfan/Dropbox/JointAttention/node_weight/validate/'
image_path='/home/lfan/Dropbox/runCoAtt/rawData/images/all/'

sf='362'

file_now='00800'+'_'+sf+'.txt'
img_now=file_now[:-4]+'.jpg'

#files=[f for f in listdir(data_path+sf) if isfile(data_path+sf+'/'+f)]

#for i in range(len(files)):
    #file_now=files[i]
    #img_now=file_now[:-4]+'.jpg'
img=cv2.imread(image_path+img_now)

with open(data_path+sf+'/'+file_now,'r') as to_read:
    lines=to_read.readlines()
for j in range(len(lines)):
    line_now_list=lines[j].split()
    x_min=int(line_now_list[0])
    y_min=int(line_now_list[1])
    x_max=int(line_now_list[2])
    y_max=int(line_now_list[3])

    cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,0,255))
    print([(x_min+x_max)*1./2,(y_min+y_max)*1./2])
    print([int((x_min+x_max)*1./2),int((y_min+y_max)*1./2)])
    loc=(int((x_min+x_max)*1./2),int((y_min+y_max)*1./2))
    cv2.putText(img,line_now_list[4]+' '+line_now_list[5],loc, cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1,cv2.LINE_AA)


cv2.imshow(file_now[:-4],img)
cv2.waitKey(0)
cv2.destroyAllWindows()

