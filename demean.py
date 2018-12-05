#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:38:39 2017

@author: tzhou
"""

#calculate mean
import skimage
import skimage.io
import skimage.transform

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#imname = '/home/tzhou/Desktop/0.png'  
#im = skimage.io.imread(imname)
#imcrop = skimage.img_as_ubyte(skimage.transform.resize(im, [224, 224]))
#imgplot = plt.imshow(imcrop)
#gray = rgb2gray(imcrop)
#imgplot = plt.imshow(gray)
#%%
a = []
i_size = 112
s_train = np.zeros((70000,i_size,i_size))
for i in range(70000):
    print(i)
    imname = '/home/tzhou/Desktop/cat/trainpng/' + str(i) +'.png'    
    im = skimage.io.imread(imname)
    #plt.imsave('test1.png', im, cmap = plt.cm.gray)  
    #imgplot = plt.imshow(im)
    imcrop = skimage.img_as_ubyte(skimage.transform.resize(im, [i_size, i_size]))
    gray = rgb2gray(imcrop)
    a.append(np.mean(gray))
    s_train[i,:,:] = gray
print(np.mean(a))
np.save('train.npy', s_train)

#%%
import numpy as np
train = np.load('train.npy')
valid = np.load('valid.npy')
test = np.load('test.npy')
np.savez('cat', train = train, valid = valid, test = test )
#%%

#s_test = np.zeros((300,224,224))
#for i in range(300):
#    print(i)
#    imname = '/home/tzhou/Desktop/sheep/test_png/' + str(i) +'.png'    
#    im = skimage.io.imread(imname)
#    #plt.imsave('test1.png', im, cmap = plt.cm.gray)  
#    #imgplot = plt.imshow(im)
#    imcrop = skimage.img_as_ubyte(skimage.transform.resize(im, [224, 224]))
#    gray = rgb2gray(imcrop)
#    s_test[i,:,:] = gray
