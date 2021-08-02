import os
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

ca_annot_path='/home/lfan/Dropbox/JointAttention/Data/annotation/'
face_train_path='/home/lfan/Dropbox/JointAttention/faces/validate_union/'
face_train_sf=[f for f in sorted(listdir(face_train_path)) if isdir(join(face_train_path,f))]

weight_train_path='/home/lfan/Dropbox/JointAttention/node_weight/validate/'

iou_threshold=0.5
inside_threshold=0.5

# face2 is almost inside face1
def almost_inside(face1,face2):

    start_x=min(face1[0],face2[0])
    start_y=min(face1[1],face2[1])
    end_x=max(face1[2],face2[2])
    end_y=max(face1[3],face2[3])

    face1_w=face1[2]-face1[0]
    face1_h=face1[3]-face1[1]

    face2_w=face2[2]-face2[0]
    face2_h=face2[3]-face2[1]

    width=face1_w+face2_w-(end_x-start_x)
    height=face1_h+face2_h-(end_y-start_y)

    if width<=0 or height<=0:
        perc=0
    else:
        perc=width*height*1./(face2_w*face2_h)
    return perc


def compute_overlap(face1,face2):

    start_x=min(face1[0],face2[0])
    start_y=min(face1[1],face2[1])
    end_x=max(face1[2],face2[2])
    end_y=max(face1[3],face2[3])

    face1_w=face1[2]-face1[0]
    face1_h=face1[3]-face1[1]

    face2_w=face2[2]-face2[0]
    face2_h=face2[3]-face2[1]

    width=face1_w+face2_w-(end_x-start_x)
    height=face1_h+face2_h-(end_y-start_y)

    if width<=0 or height<=0:
        iou=0
    else:
        iou=width*height*1./(face1_w*face1_h+face2_w*face2_h-width*height)
    return iou


for i in range(len(face_train_sf)):
    sf=face_train_sf[i]
    with open(join(ca_annot_path,sf)+'/coattention.txt','r') as ca_file_to_read:
        ca_lines=ca_file_to_read.readlines()
    #ca_lines=np.asarray(ca_lines)

    files=[f for f in sorted(listdir(join(face_train_path,sf))) if isfile(join(face_train_path,sf)+'/'+f)]

    for j in range(len(files)):
        file_now=files[j]
        frame=file_now.split('_')[0]
        with open(join(face_train_path,sf)+'/'+file_now,'r') as file_now_to_read:
            faces_union=file_now_to_read.readlines()

        ca_frame_now=[]
        for k in range(len(ca_lines)):
            ca_list=ca_lines[k].split()
            if int(ca_list[1])==int(frame):
               ca_frame_now.append(ca_lines[k])

        ca_frame_now_len=len(ca_frame_now)

        ca_frame_now_max_width=0
        for r in range(ca_frame_now_len):
            if len(ca_frame_now[r].split())>ca_frame_now_max_width:
                ca_frame_now_max_width=len(ca_frame_now[r].split())

        if ca_frame_now_len>0:
           ca_frame_now_used=np.zeros((ca_frame_now_len,(ca_frame_now_max_width-2)/4))

        if not isdir(join(weight_train_path,sf)):
            os.mkdir(join(weight_train_path,sf))

        with open(join(weight_train_path,sf)+'/'+file_now,'w') as weight_to_write:

             for q in range(len(faces_union)):
                 face_line=faces_union[q]
                 face_list=face_line.split()[3:7]
                 face=np.asarray(face_list,dtype=float)

                 ca_flag='0'
                 att_flag='0'

                 ca_cnt=0
                 for m in range(len(ca_frame_now)):
                     ca_cnt=ca_cnt+1
                     ca_now_list=ca_frame_now[m].split()

                     ca_node = ca_now_list[2:6]
                     ca_node = np.asarray(ca_node, dtype=float)
                     iou = compute_overlap(face, ca_node)
                     perc=almost_inside(face,ca_node)
                     if iou>iou_threshold:
                         ca_flag=str(ca_cnt)
                         ca_frame_now_used[ca_cnt-1,0]=1
                     elif perc>inside_threshold:
                         ca_flag = str(ca_cnt)
                         ca_frame_now_used[ca_cnt - 1, 0] = 1

                     for n in range(0, (len(ca_now_list) - 6) / 4):
                         ca_node=ca_now_list[(n*4+6):(n*4+10)]
                         ca_node=np.asarray(ca_node,dtype=float)

                         iou=compute_overlap(face, ca_node)
                         if iou>iou_threshold:
                             att_flag=str(ca_cnt)
                             ca_frame_now_used[ca_cnt-1,n+1]=1

                 weight_to_write.write(face_list[0] + ' ' + face_list[1] + ' ' + face_list[2] + ' ' + face_list[3] +' '+ca_flag + ' '+att_flag + '\n')

             ca_cnt=0
             for p in range(len(ca_frame_now)):
                 ca_cnt=ca_cnt+1
                 ca_now_list=ca_frame_now[p].split()

                 if ca_frame_now_used[ca_cnt-1,0]==0:
                     ca_node=ca_now_list[2:6]
                     weight_to_write.write(ca_node[0]+' '+ca_node[1]+' '+ca_node[2]+' '+ca_node[3]+' '+str(ca_cnt)+' '+'0\n')

                 for s in range(0, (len(ca_now_list) - 6) / 4):
                     if ca_frame_now_used[ca_cnt-1,s+1]==0:
                        ca_node = ca_now_list[(s * 4 + 6):(s * 4 + 10)]
                        weight_to_write.write(ca_node[0] + ' ' + ca_node[1] + ' ' + ca_node[2] + ' ' + ca_node[3] + ' ' + '0' + ' ' + str(ca_cnt)+'\n')



