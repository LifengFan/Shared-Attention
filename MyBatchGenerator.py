import numpy as np
from scipy.misc import imread

def TwoStream_generator(batch_size,x1_path,x2_path,y_path,nb_classes):
# batch_size = 1
# nb_classes = 2
# #
# x1_path = '/home/yang/Desktop/ICCV2017/Data/TPV_Frames/KerasRGBFluents/c14_open_door/Door_Train.txt'
# x2_path = '/home/yang/Desktop/ICCV2017/Data/TPV_Frames/KerasRGBFluents/c14_open_door/FullImg_Train.txt'
# y_path = '/home/yang/Desktop/ICCV2017/Data/TPV_Frames/KerasRGBFluents/c14_open_door/Cate_Train.txt'
    x1_file = np.genfromtxt(x1_path, dtype=None)
    x2_file = np.genfromtxt(x2_path, dtype=None)
    y_file = np.genfromtxt(y_path)
    while 1:
        for i in range(0,len(x1_file)//batch_size):       # floor divide
            x1_batch = np.zeros(shape=(batch_size,3,224,224))
            x2_batch = np.zeros(shape=(batch_size,3,224,224))
            y_batch = np.zeros(shape=(batch_size,nb_classes))
            for j in range(0,batch_size):
                im1 = imread(x1_file[i*batch_size+j])
                im2 = imread(x2_file[i*batch_size+j])
                im1 = im1.astype('float32')
                im2 = im2.astype('float32')
                im1[:, :, 0] -= 123.68
                im1[:, :, 1] -= 116.779
                im1[:, :, 2] -= 103.939
                im1 = im1.transpose((2, 0, 1))
                im2[:, :, 0] -= 123.68
                im2[:, :, 1] -= 116.779
                im2[:, :, 2] -= 103.939
                im2 = im2.transpose((2, 0, 1))
                label = y_file[i*batch_size+j]
                x1_batch[j] = im1
                x2_batch[j] = im2
                y_batch[j] = label
            #print('One Batch')
            yield [x1_batch, x2_batch], y_batch


def Img_From_List(batch_size, x_path, y_path, nb_classes):
    x_file = np.genfromtxt(x_path, dtype=None)
    y_file = np.genfromtxt(y_path)
    while 1:
        for i in range(0, len(x_file) // batch_size):  # floor divide
            x_batch = np.zeros(shape=(batch_size, 3, 224, 224))
            y_batch = np.zeros(shape=(batch_size, nb_classes))
            for j in range(0, batch_size):
                # print (x_file[i * batch_size + j])
                im = imread(x_file[i * batch_size + j])
                im = im.astype('float32')
                im[:, :, 0] -= 123.68
                im[:, :, 1] -= 116.779
                im[:, :, 2] -= 103.939
                im = im.transpose((2, 0, 1))
                label = y_file[i * batch_size + j]
                x_batch[j] = im
                y_batch[j] = label
            # print('One Batch')
            yield x_batch, y_batch