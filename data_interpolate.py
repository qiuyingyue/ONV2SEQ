#!/usr/bin/env python
# coding: utf-8

# In[2]:



# import the required libraries# impor 
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)


# In[3]:


tf.logging.info("TensorFlow Version: %s", tf.__version__)


# In[4]:



# import our command line tools# impor 
from sketch_rnn_train import *
from model import *
from utils import *
from rnn import *


# In[5]:


# little function that displays vector images and saves them to .svg
img_default_dir = 'sample.svg'
def draw_strokes(data, factor=0.5, svg_filename = img_default_dir):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  #display(SVG(dwg.tostring()))


# In[6]:


def generate_np_arr(data, size_bound = 600, factor=1.0):
    arr = []
    x_pos = 0
    y_pos = 0


    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    width= max_x-min_x
    height= max_y-min_y
    size = max(width, height)
    factor =  float(size)/size_bound


    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    print factor
    for i in range(len(data)):
        point = data[i]
        deltax = float(point[0])/factor
        deltay = float(point[1])/factor
        pen_lift = point[2]
        
        x_pos+=deltax
        min_x = min(x_pos, min_x)
        max_x = max(x_pos, max_x)
        y_pos+=deltay
        min_y = min(-y_pos, min_y)
        max_y = max(-y_pos, max_y)
        arr.append([x_pos,-y_pos, pen_lift])
    arr = np.asarray(arr)
    print min_x,min_y, max_x, max_y
    start_x = 0
    start_y = 0
    print start_y-min_y
    arr[:,0] = np.add(arr[:,0],start_x-min_x)
    arr[:,1] = np.add(arr[:,1],start_y-min_y)
    return arr


# In[7]:



def add_data_point(my_data,offset_x=0,offset_y=0):
    new_data=[]
    for i in range(0, len(my_data)-1):
        point_x = my_data[i:i+2,0]
        point_y = my_data[i:i+2,1]
        point_x = np.add(point_x,offset_x)
        point_y = np.add(point_y,offset_y)
        abs_x = np.abs(point_x[0]-point_x[1])
        abs_y = np.abs(point_y[0]-point_y[1])
        dis = np.int(max(abs_x,abs_y))
        new_x_arr = np.linspace(point_x[0],point_x[1],dis)
        new_y_arr = np.linspace(point_y[0],point_y[1],dis)
        new_x_arr = np.round(new_x_arr,2)
        new_y_arr = np.round(new_y_arr,2)
        for j in range(0,dis):
            new_data.append([new_x_arr[j],new_y_arr[j],my_data[i,2]])

    new_data = np.asarray(new_data)
    return new_data

# In[9]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')

  # input stroke
  file_path = '../data/sketch/sketchrnn_cat.full.npz'
  data = np.load(file_path)
  train_set = data['train']
  valid_set = data['valid']
  test_set = data['test']

  train_set_idx = 200
  stroke = train_set[train_set_idx]

  # display the original image for comparison
  draw_strokes(stroke,factor=1, svg_filename=img_default_dir)

  # generate data point coordinates based on stroke sequences
  arr= generate_np_arr(stroke, size_bound=600)

  row1 = arr[0,:]
  row1 = row1.reshape(1,3)
  row1[:,2]=1
  print row1
  arr=np.append(arr,row1,axis=0)
  print arr.shape

  # intepolate data points
  new_data = add_data_point(arr)

  # display image with new data points
  for i in range(0, len(new_data)-1):
      if new_data[i,2] ==0:
          plt.plot(new_data[i:i+2,0],new_data[i:i+2,1])

  plt.show()

  # save data points to file
  save_file_name = 'cat.csv'
  np.savetxt(save_file_name, new_data, delimiter=",")




