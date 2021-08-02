import numpy as np

def fill(heatmap):
    """
Flood fill algorithm

Parameters
----------
data : (M, N) ndarray of uint8 type
    Image with flood to be filled. Modified inplace.

Returns
-------
List, each component represents a connected area, which contains the position of each point.
"""

    region = []
    heatmap_t = heatmap.copy()
    (xsize, ysize) = heatmap_t.shape
    ini = 2

    while True:
        if 1 in heatmap_t:
            tmp_region = []
            stack = []
            value_index = np.where(heatmap_t == 1)
            cord = [value_index[0][0], value_index[1][0]]
            stack.append(cord)
            # stack = set(cord)
            while stack:
                x, y = stack.pop()
                if heatmap_t[x, y] == 1:
                    tmp_region.append((x, y))
                    heatmap_t[x, y] = ini
                    if x > 0:
                        stack.append((x - 1, y))
                    if x < (xsize - 1):
                        stack.append((x + 1, y))
                    if y > 0:
                        stack.append((x, y - 1))
                    if y < (ysize - 1):
                        stack.append((x, y + 1))
            region.append(tmp_region)
            ini = ini + 1
        else:
            break

    return region, heatmap_t




