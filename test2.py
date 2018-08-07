# coding: utf-8

import keras
from keras.models import load_model
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import cv2
import numpy as np
import h5py
import os

# model = Sequential()
model = load_model('/usr/code/NN/captcha-recognition-master/Model/my_model_test1.h5')


#x_test准备
import csv

csvfile = open('/usr/code/NN/captcha-recognition-master/images/4-c-1-g/train/lables.csv')
reader = csv.reader(csvfile)

lables = []
for line in reader:
    #print line
    tmp = [line[0],line[1]]
    #print tmp
    lables.append(tmp)

csvfile.close()



X = []#放序号图片矩阵，序号找文件名
picnum = len(lables) #总图片数
print "picnum : ",picnum
for i in range(0,6000):
    img = cv2.imread("images/4-c-1-g/train/" + lables[i][0] + '.png', cv2.IMREAD_GRAYSCALE)
    X.append(img)


# In[22]:

print len(X),X[0].shape


X = np.array(X)

print X.shape

x_test = X[4000:6000]

img_rows, img_cols = 60, 80
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print(x_test.shape)



x_test = 255 - x_test
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')



### In[35]:

pred = model.predict_classes(x_test,batch_size = 32,verbose=0)
count=0
print '\n'
print pred
for i in pred:
    if (i==[52]):
        count+=1
print count



