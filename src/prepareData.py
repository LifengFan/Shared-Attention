import os
from os import listdir
from os.path import isfile, join
import cv2

# ## prepare training data
# trainAnnotPath='/home/lfan/Dropbox/runCoAtt/rawData/annotation/train/'
# trainVideoPath='/home/lfan/Dropbox/runCoAtt/rawData/videos/train/'
# trainAnnotData=[f for f in listdir(trainAnnotPath) if isfile(join(trainAnnotPath,f))]
# trainVideoData=[f for f in listdir(trainVideoPath) if isfile(join(trainVideoPath,f))]

# # video to image
# for id in range(len(trainVideoData)):
#    filename = trainVideoPath + trainVideoData[id]
#    vid=trainVideoData[id][:-4]
#    os.system('ffmpeg -i '+filename+ ' /home/lfan/Dropbox/runCoAtt/rawData/images/train/%5d_'+vid+'.jpg')


#get label .txt file
# for id in range(len(trainAnnotData)):
#    filename=trainAnnotPath+trainAnnotData[id]
#    vid=trainAnnotData[id][6:-4]
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
#          if label_tmp[:-1]!='person' and label_tmp[:-1]!='face' and label_tmp[:-2]!='person' and label_tmp[:-2]!='face':
#             continue
#          if label_tmp[:-1]=='person' or label_tmp[:-1]=='face':
#             label_tmp=label_tmp[:-1]
#          else:
#             label_tmp = label_tmp[:-2]
#
#          frame_tmp=str(int(frame_tmp)+1)
#          label4img='%5s_%s.txt'%(frame_tmp.zfill(5),vid)
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
#          if label_tmp=='person':
#             label_num='0'
#          elif label_tmp=='face':
#             label_num='1'
#          with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/train/images/'+label4img,'a+') as f:
#             f.write(label_num+" "+str(x_tmp)+" "+str(y_tmp)+" "+str(w_tmp)+" "+str(h_tmp)+'\n')
#
#          #os.system('cp /home/lfan/Dropbox/runCoAtt/rawData/images/train/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid)+' /home/lfan/Dropbox/runCoAtt/darknet/mydata/train/images/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid))
#
#          #img=cv2.imread('/home/lfan/Dropbox/runCoAtt/rawData/images/train/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid))
#          #cv2.imwrite('/home/lfan/Dropbox/runCoAtt/darknet/mydata/train/images/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid),img)


# mypath='/home/lfan/Dropbox/runCoAtt/darknet/mydata/train/images/'
# allfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#
# print(len(allfiles))
#
# with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/train.txt','w') as f:
#    for i in range(len(allfiles)):
#       f.write(mypath+allfiles[i]);
#       f.write('\n')


# ## prepare validation data
validAnnotPath='/home/lfan/Dropbox/runCoAtt/rawData/annotation/validate/'
validVideoPath='/home/lfan/Dropbox/runCoAtt/rawData/videos/validate/'
validAnnotData=[f for f in listdir(validAnnotPath) if isfile(join(validAnnotPath,f))]
validVideoData=[f for f in listdir(validVideoPath) if isfile(join(validVideoPath,f))]

# # video to image
# for id in range(len(validVideoData)):
#    filename = validVideoPath + validVideoData[id]
#    vid=validVideoData[id][:-4]
#    os.system('ffmpeg -i '+filename+ ' /home/lfan/Dropbox/runCoAtt/rawData/images/validate/%5d_'+vid+'.jpg')

# #get label .txt file
# for id in range(len(validAnnotData)):
#    filename=validAnnotPath+validAnnotData[id]
#    vid=validAnnotData[id][6:-4]
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
#          if label_tmp[:-1]!='person' and label_tmp[:-1]!='face' and label_tmp[:-2]!='person' and label_tmp[:-2]!='face':
#             continue
#          if label_tmp[:-1]=='person' or label_tmp[:-1]=='face':
#             label_tmp=label_tmp[:-1]
#          else:
#             label_tmp = label_tmp[:-2]
#
#          frame_tmp=str(int(frame_tmp)+1)
#          label4img='%5s_%s.txt'%(frame_tmp.zfill(5),vid)
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
#          if label_tmp=='person':
#             label_num='0'
#          elif label_tmp=='face':
#             label_num='1'
#          with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/validate/images/'+label4img,'a+') as f:
#             f.write(label_num+" "+str(x_tmp)+" "+str(y_tmp)+" "+str(w_tmp)+" "+str(h_tmp)+'\n')

         #os.system('cp /home/lfan/Dropbox/runCoAtt/rawData/images/validate/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid)+' /home/lfan/Dropbox/runCoAtt/darknet/mydata/validate/images/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid))

         #img=cv2.imread('/home/lfan/Dropbox/runCoAtt/rawData/images/train/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid))
         #cv2.imwrite('/home/lfan/Dropbox/runCoAtt/darknet/mydata/train/images/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid),img)

# mypath='/home/lfan/Dropbox/runCoAtt/darknet/mydata/validate/images/'
# allfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#
# print(len(allfiles))
#
# with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/validate.txt','w') as f:
#    for i in range(len(allfiles)):
#       f.write(mypath+allfiles[i]);
#       f.write('\n')
#

# ## prepare testing data
testAnnotPath='/home/lfan/Dropbox/runCoAtt/rawData/annotation/test/'
testVideoPath='/home/lfan/Dropbox/runCoAtt/rawData/videos/test/'
testAnnotData=[f for f in listdir(testAnnotPath) if isfile(join(testAnnotPath,f))]
testVideoData=[f for f in listdir(testVideoPath) if isfile(join(testVideoPath,f))]
#
# # video to image
# for id in range(len(testVideoData)):
#    filename = testVideoPath + testVideoData[id]
#    vid=testVideoData[id][:-4]
#    os.system('ffmpeg -i '+filename+ ' /home/lfan/Dropbox/runCoAtt/rawData/images/test/%5d_'+vid+'.jpg')


#get label .txt file
for id in range(len(testAnnotData)):
   filename=testAnnotPath+testAnnotData[id]
   vid=testAnnotData[id][6:-4]

   with open(filename,'r') as file_to_read:
      while True:
         lines=file_to_read.readline()
         lines=lines[:-1]
         if len(lines)==0:
            break
         trackID_tmp,xmin_tmp,ymin_tmp,xmax_tmp,ymax_tmp,frame_tmp,lost_tmp,occluded_tmp,generated_tmp,label_tmp=lines.split(' ',9)
         if int(lost_tmp)==1 or int(occluded_tmp)==1:
            continue

         label_tmp=label_tmp[1:-1]
         if label_tmp[:-1]!='person' and label_tmp[:-1]!='face' and label_tmp[:-2]!='person' and label_tmp[:-2]!='face':
            continue
         if label_tmp[:-1]=='person' or label_tmp[:-1]=='face':
            label_tmp=label_tmp[:-1]
         else:
            label_tmp = label_tmp[:-2]

         frame_tmp=str(int(frame_tmp)+1)
         label4img='%5s_%s.txt'%(frame_tmp.zfill(5),vid)
         xmin_tmp = float(xmin_tmp)
         xmax_tmp = float(xmax_tmp)
         ymin_tmp = float(ymin_tmp)
         ymax_tmp = float(ymax_tmp)
         dw_tmp=1./480
         dh_tmp=1./320
         x_tmp = (xmin_tmp + xmax_tmp) / 2.0 - 1
         y_tmp = (ymin_tmp + ymax_tmp) / 2.0 - 1
         x_tmp = x_tmp * dw_tmp
         y_tmp = y_tmp * dh_tmp
         w_tmp = (xmax_tmp-xmin_tmp) * dw_tmp
         h_tmp = (ymax_tmp-ymin_tmp) * dh_tmp
         if label_tmp=='person':
            label_num='0'
         elif label_tmp=='face':
            label_num='1'
         with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/test/images/'+label4img,'a+') as f:
            f.write(label_num+" "+str(x_tmp)+" "+str(y_tmp)+" "+str(w_tmp)+" "+str(h_tmp)+'\n')

         #os.system('cp /home/lfan/Dropbox/runCoAtt/rawData/images/test/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid)+' /home/lfan/Dropbox/runCoAtt/darknet/mydata/test/images/'+'%5s_%s.jpg'%(frame_tmp.zfill(5),vid))

#
#
# mypath='/home/lfan/Dropbox/runCoAtt/darknet/mydata/test/images/'
# allfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#
# print(len(allfiles))
#
# with open('/home/lfan/Dropbox/runCoAtt/darknet/mydata/test.txt','w') as f:
#    for i in range(len(allfiles)):
#       f.write(mypath+allfiles[i]);
#       f.write('\n')
