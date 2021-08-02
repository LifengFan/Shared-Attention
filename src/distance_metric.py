import numpy as np
import floodfill
import cv2
import matplotlib.pyplot as plt

RANGE = 50

def getGazeHeatmap(heatmap, thre):
    (n,m) = heatmap.shape
    tmp = np.zeros((n,m))
    tmp[heatmap > thre] = 1

    return tmp



def getDistance1(heatmap,groundtruth,thre):
    distance = np.zeros((1,RANGE))

    tmp = getGazeHeatmap(heatmap, thre)
    dis_t = []

    region, ht = floodfill.fill(tmp)
    pred_center = []
    gt_center = []
    if len(region) != 0:
        for iter in range(len(region)):
            a = np.asarray(region[iter])
            x = sum(a[:, 0]) / float(len(a[:, 0]))
            y = sum(a[:, 1]) / float(len(a[:, 1]))
            pred_center.append([x, y])
    pred = np.asarray(pred_center)
    gt, gt_ht = floodfill.fill(groundtruth)
    if len(gt) != 0:
        for iter in range(len(gt)):
            a = np.asarray(gt[iter])
            x = sum(a[:, 0]) / float(len(a[:, 0]))
            y = sum(a[:, 1]) / float(len(a[:, 1]))
            gt_center.append([x, y])

    if len(gt_center) == 0:
        return distance
    else:
        while gt_center:
            if len(pred) == 0:
                for l in range(len(gt_center)):
                    dis_t.append(420) #if there's no potential predicted point,return 420
                break
            a = gt_center.pop()
        # for jter in range(len(gt_center)):
        #     a = np.asarray(gt_center[jter])
            b = pred - a
            diff = np.sqrt(np.square(b[:,0]) + np.square(b[:,1]))
            pos = diff.argmin()
            dis_t.append(diff.min())
            pred = np.delete(pred, (0), axis=0)

    return float(sum(dis_t))

def getDistance(heatmap, groundtruth):
    distance = np.zeros((1, RANGE))

    for i in range(RANGE):
        thre = float(1) / float(RANGE) * float(i + 1)
        tmp = getGazeHeatmap(heatmap, thre)
        dis_t = []

        region, ht = floodfill.fill(tmp)
        pred_center = []
        gt_center = []
        if len(region) != 0:
            for iter in range(len(region)):
                a = np.asarray(region[iter])
                x = sum(a[:, 0]) / float(len(a[:, 0]))
                y = sum(a[:, 1]) / float(len(a[:, 1]))
                pred_center.append([x, y])
        pred = np.asarray(pred_center)
        gt, gt_ht = floodfill.fill(groundtruth)
        if len(gt) != 0:
            for iter in range(len(gt)):
                a = np.asarray(gt[iter])
                x = sum(a[:, 0]) / float(len(a[:, 0]))
                y = sum(a[:, 1]) / float(len(a[:, 1]))
                gt_center.append([x, y])

        if len(gt_center) == 0:
            return distance
        else:
            while gt_center:
                if len(pred) == 0:
                    for l in range(len(gt_center)):
                        dis_t.append(420)  # if there's no potential predicted point,return 420
                    break
                a = gt_center.pop()
                # for jter in range(len(gt_center)):
                #     a = np.asarray(gt_center[jter])
                b = pred - a
                diff = np.sqrt(np.square(b[:, 0]) + np.square(b[:, 1]))
                pos = diff.argmin()
                dis_t.append(diff.min())
                pred = np.delete(pred, (0), axis=0)

        distance[0, i] = float(sum(dis_t))
    return distance









    # plt.figure()
    #
    # plt.subplot(1, 2, 1)
    # plt.title('original heatmap')
    # plt.imshow(tmp)
    #
    # plt.subplot(1, 2, 2)
    # plt.title('clustered heatmap')
    # plt.imshow(ht)
    # plt.axis('off')
    #
    # plt.show()



# heatmap = np.loadtxt('hm.txt')
# thre = 1.2
# groundtruth = tmp = getGazeHeatmap(heatmap, 1.7)
# dis = getDistance(heatmap, thre, groundtruth)
# print(dis)