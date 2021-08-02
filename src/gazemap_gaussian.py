import math
import numpy as np
def getGazeHeatmap(face_loc, gaze_dir):
    sigma = 0.5
    heatmap = np.zeros(shape=(16, 24))
    dx = gaze_dir[0]
    dy = gaze_dir[1]
    fx = face_loc[0]
    fy = face_loc[1]

    for i in range(24):
        for j in range(16):
            theta = math.acos(
                ((i - fx) * dx + (j - fy) * dy) / math.sqrt((dx ** 2 + dy ** 2+0.00001) * ((i - fx) ** 2 + (j - fy) ** 2+0.00001)))
            heatmap[j, i] = (1. / sigma) * math.exp(-theta ** 2 / (2 * (sigma ** 2)))

    return heatmap