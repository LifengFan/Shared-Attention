from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt

##input: heatmap
##output:
##heatmap+boundingbox->scored box->threshold&IOU/I->(ifframe,tp,#detected,#proposal,num_ca)

gaze_dir_prefix = '/home/yixin/Dropbox/JointAttention/tested_face_direction/test/'
prop_path = '/media/yixin/Elements/bbx_proposal/'
RANGE = 1000

def eval_bbx(heatmap,list_now):

    vid = list_now[0].split('_')[-1][:-4]
    ob_bbx = []
    score = []
    gt = []
    #ini output
    ifframe = np.zeros((1,RANGE))
    tp = np.zeros((1, RANGE))
    fp = np.zeros((1, RANGE))
    true_pos = np.zeros((1,RANGE))
    detected_bbx = np.zeros((1,RANGE))
    num_ca = 0


    ##extract object bbx
    if isfile(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt'):
        with open(gaze_dir_prefix + list_now[0].split('/')[-1][:-4] + '.txt', 'r') as gaze_to_read:
            gaze_lines = gaze_to_read.readlines()

        face_tmp = np.zeros(shape=(320, 480))

        for gaze_id in range(len(gaze_lines)):
            gaze_list = gaze_lines[gaze_id].split()

            # cv2.rectangle(img_orig, (int(gaze_list[3]), int(gaze_list[4])),
            #               (int(gaze_list[5]), int(gaze_list[6])), (0, 255, 255))

            g_xmin = int(float(gaze_list[3]))
            g_ymin = int(float(gaze_list[4]))
            g_xmax = int(float(gaze_list[5]))
            g_ymax = int(float(gaze_list[6]))

            ob_bbx.append([g_ymin,g_xmin,g_ymax,g_xmax])


    if isfile(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt'):

        # object_tmp = np.zeros(shape=(320, 480))

        with open(prop_path + vid + '/' + str(int(list_now[0].split('/')[-1].split('_')[0])) + '.txt',
                  'r') as f2:
            while True:
                line = f2.readline()
                if len(line) == 0:
                    break
                list = line.split()
                xmin = float(list[0])
                ymin = float(list[1])
                xmax = float(list[2])
                ymax = float(list[3])

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)

                ob_bbx.append([xmin, ymin, xmax, ymax])
    ##score for bbx
    for i in range(len(ob_bbx)):
        a = ob_bbx[i]
        submap = heatmap[a[1]:a[3],a[0]:a[2]]
        score.append(submap.sum()/((a[2]-a[0])*(a[3]-a[1])))


    # max_value = max(score)
    # t = np.array(0.2, dtype=np.float64)
    # max_index = score.index(max_value)
    # score = score/t

    #extract co-attention

    if len(list_now) > 1:

        num_ca = (len(list_now) - 1) / 4
        for ca_id in range(num_ca):
            xmin = float(list_now[4 * ca_id + 1])
            ymin = float(list_now[4 * ca_id + 2])
            xmax = float(list_now[4 * ca_id + 3])
            ymax = float(list_now[4 * ca_id + 4])

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            gt.append([xmin, ymin, xmax, ymax])

    ##metric 1: if co-attention is detected correctly in this frame
    score = np.asarray(score)

    for i in range(RANGE):
        judge = score >= float(1)/ float(RANGE) * float(i+1)
        judge = judge.tolist()
        if num_ca != 0:
            detected_bbx[0, i] = judge.count(True)

        if True in judge and num_ca > 0:
            ifframe[0, i] = 1
            tp[0, i] = 1
        if True in judge and num_ca == 0:
            fp[0, i] = 1
        if True not in judge and num_ca == 0:
            ifframe[0, i] = 1


    #metric 2: precision/recall
    if num_ca != 0:

        detect = np.zeros((len(ob_bbx), len(gt)))
        for i in range(len(ob_bbx)):
            a = ob_bbx[i]
            for j in range(len(gt)):
                b = gt[j]
                if bb_if_intersection(a, b):
                    detect[i,j] = 1

        for i in range(RANGE):
            judge = np.where(score >=  float(1)/ float(RANGE) * float(i+1))
            judge = judge[0]
            # print(len(gt))
            # print(len(judge))
            for l in range(len(gt)):
                for j in range(len(judge)):
                    if detect[judge[j], l] >= 1:
                        true_pos[0, i] = true_pos[0, i] + 1
                        break

    return tp, fp, ifframe, true_pos, detected_bbx, num_ca



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def bb_if_intersection(boxA, boxB):
    hoverlaps = True
    voverlaps = True
    if (boxA[0] > boxB[2]) or (boxA[2] < boxB[0]):#[0]xmin[1]ymin[2]xmax[3]ymax
        hoverlaps = False
    if (boxA[3] < boxB[1]) or (boxA[1] > boxB[3]):
        voverlaps = False
    return hoverlaps and voverlaps


