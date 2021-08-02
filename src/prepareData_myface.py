import os
from os import listdir
from os.path import isfile, join
import cv2

# trainset=[1,2,3,4,5,6,7,8,9,10,11,12,15,16,18,19,20,22,23,24,25,
#           26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,43,44,45,46,
#           47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,73,362,363,364,365,366,367,368,369,370,371]
#
# validateset=[372,373,374,375,376,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400]


# for id in range(len(trainset)):
#     vid=trainset[id]
#     videoname='/home/lfan/Dropbox/JointAttention/Data/videos/all/'+str(vid)+'.mp4'
#     os.system('ffmpeg -i '+videoname+ ' /home/lfan/Dropbox/runCoAtt/darknet/mydata2/rawimages/train/%5d_'+str(vid)+'.jpg')

# for id in range(len(validateset)):
#     vid=validateset[id]
#     videoname='/home/lfan/Dropbox/JointAttention/Data/videos/all/'+str(vid)+'.mp4'
#     os.system('ffmpeg -i '+videoname+ ' /home/lfan/Dropbox/runCoAtt/darknet/mydata2/rawimages/validate/%5d_'+str(vid)+'.jpg')



# for id in range(len(trainset)):
#    vid=trainset[id]
#
#    filename='/home/lfan/Dropbox/JointAttention/Data/rawAnnotation/'+'output'+str(vid)+'.txt'
#
#    with open(filename,'r') as file_to_read:
#       while True:
#          lines=file_to_read.readline()
#          lines=lines[:-1]
#          if len(lines)==0:
#             break
#          trackID_tmp,xmin_tmp,ymin_tmp,xmax_tmp,ymax_tmp,frame_tmp,lost_tmp,occluded_tmp,generated_tmp,label_tmp=lines.split(' ',9)
#          if int(lost_tmp)==1 or int(occluded_tmp)==1:
#             continue
#
#          label_tmp=label_tmp[1:-1]
#          if label_tmp[0:4]!='face':
#              continue
#
#          frame_tmp=str(int(frame_tmp)+1)
#          if int(frame_tmp)>300:
#              continue
#
#          label4img='%5s_%s.txt'%(frame_tmp.zfill(5),str(vid))
#          xmin_tmp = float(xmin_tmp)
#          xmax_tmp = float(xmax_tmp)
#          ymin_tmp = float(ymin_tmp)
#          ymax_tmp = float(ymax_tmp)
#          dw_tmp=1./480
#          dh_tmp=1./320
#          x_tmp = (xmin_tmp + xmax_tmp) / 2.0 - 1
#          y_tmp = (ymin_tmp + ymax_tmp) / 2.0 - 1
#          x_tmp = x_tmp * dw_tmp
#          y_tmp = y_tmp * dh_tmp
#          w_tmp = (xmax_tmp-xmin_tmp) * dw_tmp
#          h_tmp = (ymax_tmp-ymin_tmp) * dh_tmp
#
#          with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata2/train/labels/'+label4img,'a+') as f:
#             f.write('0'+" "+str(x_tmp)+" "+str(y_tmp)+" "+str(w_tmp)+" "+str(h_tmp)+'\n')



# for id in range(len(validateset)):
#    vid=validateset[id]
#
#    filename='/home/lfan/Dropbox/JointAttention/Data/rawAnnotation/'+'output'+str(vid)+'.txt'
#
#    with open(filename,'r') as file_to_read:
#       while True:
#          lines=file_to_read.readline()
#          lines=lines[:-1]
#          if len(lines)==0:
#             break
#          trackID_tmp,xmin_tmp,ymin_tmp,xmax_tmp,ymax_tmp,frame_tmp,lost_tmp,occluded_tmp,generated_tmp,label_tmp=lines.split(' ',9)
#          if int(lost_tmp)==1 or int(occluded_tmp)==1:
#             continue
#
#          label_tmp=label_tmp[1:-1]
#          if label_tmp[0:4]!='face':
#              continue
#
#          frame_tmp=str(int(frame_tmp)+1)
#          if int(frame_tmp)>300:
#              continue
#
#          label4img='%5s_%s.txt'%(frame_tmp.zfill(5),str(vid))
#          xmin_tmp = float(xmin_tmp)
#          xmax_tmp = float(xmax_tmp)
#          ymin_tmp = float(ymin_tmp)
#          ymax_tmp = float(ymax_tmp)
#          dw_tmp=1./480
#          dh_tmp=1./320
#          x_tmp = (xmin_tmp + xmax_tmp) / 2.0 - 1
#          y_tmp = (ymin_tmp + ymax_tmp) / 2.0 - 1
#          x_tmp = x_tmp * dw_tmp
#          y_tmp = y_tmp * dh_tmp
#          w_tmp = (xmax_tmp-xmin_tmp) * dw_tmp
#          h_tmp = (ymax_tmp-ymin_tmp) * dh_tmp
#
#          with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata2/validate/labels/'+label4img,'a+') as f:
#             f.write('0'+" "+str(x_tmp)+" "+str(y_tmp)+" "+str(w_tmp)+" "+str(h_tmp)+'\n')

# train_label_path='/home/lfan/Dropbox/runCoAtt/darknet/mydata2/train/labels/'
# train_label_names=[f for f in listdir(train_label_path) if isfile(join(train_label_path,f))]

# for i in range(len(train_label_names)):
#     name=train_label_names[i]
#     vid=name[6:-4]
#     frame=name[0:5]
#     os.system('cp /home/lfan/Dropbox/runCoAtt/darknet/mydata2/rawimages/train/'+'%5s_%s.jpg'%(frame.zfill(5),vid)+' /home/lfan/Dropbox/runCoAtt/darknet/mydata2/train/images/'+'%5s_%s.jpg'%(frame.zfill(5),vid))

# validate_label_path='/home/lfan/Dropbox/runCoAtt/darknet/mydata2/validate/labels/'
# validate_label_names=[f for f in listdir(validate_label_path) if isfile(join(validate_label_path,f))]
#
#
# for i in range(len(validate_label_names)):
#     name=validate_label_names[i]
#     vid=name[6:-4]
#     frame=name[0:5]
#     os.system('cp /home/lfan/Dropbox/runCoAtt/darknet/mydata2/rawimages/validate/'+'%5s_%s.jpg'%(frame.zfill(5),vid)+' /home/lfan/Dropbox/runCoAtt/darknet/mydata2/validate/images/'+'%5s_%s.jpg'%(frame.zfill(5),vid))


# train_image_path='/home/lfan/Dropbox/runCoAtt/darknet/mydata2/train/images/'
# train_image_names=[join(train_image_path,f) for f in listdir(train_image_path) if isfile(join(train_image_path,f))]

# with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata2/train.txt','w') as file_to_write:
#     for i in range(len(train_image_names)):
#         name=train_image_names[i]
#         file_to_write.write(name)
#         file_to_write.write('\n')

# validate_image_path='/home/lfan/Dropbox/runCoAtt/darknet/mydata2/validate/images/'
# validate_image_names=[join(validate_image_path,f) for f in listdir(validate_image_path) if isfile(join(validate_image_path,f))]
#
#
# with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata2/validate.txt','w') as file_to_write:
#     for i in range(len(validate_image_names)):
#         name=validate_image_names[i]
#         file_to_write.write(name)
#         file_to_write.write('\n')
#


# test_images_path='/home/lfan/Dropbox/runCoAtt/rawData/images/test/'
#
# test_images_files=[join(test_images_path,f) for f in listdir(test_images_path) if isfile(join(test_images_path,f))]
#
# with open('/home/lfan/Dropbox/runCoAtt/rawData/images/test.txt','w') as file_to_write:
#     for i in range(len(test_images_files)):
#
#         file_to_write.write(test_images_files[i])
#         file_to_write.write('\n')

validate_annot_path='/home/lfan/Dropbox/runCoAtt/rawData/annotation/validate/'
validate_annot_files=[join(validate_annot_path,f) for f in listdir(validate_annot_path) if isfile(join(validate_annot_path,f))]

for i in range(len(validate_annot_files)):
    filename=validate_annot_files[i]
    splitlist=filename.split('output')
    vid=splitlist[1][:-4]
    with open(filename,'r') as file_to_read:
      while True:
        line=file_to_read.readline()
        line=line[:-1]
        if len(line)==0:
            break
        trackID_tmp, xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp, frame_tmp, lost_tmp, occluded_tmp, generated_tmp, label_tmp = line.split(' ', 9)
        if int(lost_tmp) == 1 or int(occluded_tmp) == 1:
            continue

        label_tmp = label_tmp[1:-1]
        if label_tmp[0:4] != 'face':
            continue

        frame_tmp = str(int(frame_tmp) + 1)

        label4img = '%5s_%s.txt' % (frame_tmp.zfill(5), vid)

        with open('/home/lfan/Dropbox/JointAttention/faces/validate/' + label4img, 'a+') as f:
            f.write('NA' + " " + 'head' + " " + 'NA' + " " + str(xmin_tmp) + " " + str(ymin_tmp)  + " " + str(xmax_tmp)+ " " + str(ymax_tmp)+'\n')











