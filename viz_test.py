import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical

image_path='/home/lfan/keras/testsample/test1/6.jpg'
top_model_weights_path='bottleneck_fc_model.h5'
orig=cv2.imread(image_path)

print("[INFO] loading and preprocessing image...")
image=load_img(image_path,target_size=(150,150))
image=img_to_array(image)

image=image/255
image=np.expand_dims(image,axis=0)

# build the VGG16 network
model=applications.VGG16(include_top=False,weights='imagenet')

# get the bottleneck prediction from the pre-trained VGG16 model
bottleneck_predictions=model.predict(image)

# build top model
model=Sequential()
model.add(Flatten(input_shape=bottleneck_predictions.shape[1:]))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.load_weights(top_model_weights_path)

# use the bottleneck prediction on the top model to get the final classification
class_predicted=model.predict_classes(bottleneck_predictions)

inID=class_predicted[0]
#class_dictionary=generator_top.class_indices


print(inID)






