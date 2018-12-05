# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
#sys.path.append("/home/tao/miniconda2/lib/python2.7/site-packages")
#sys.path.append("./code/")
# import the required libraries
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


# In[4]:


# libraries required for visualisation:
from IPython.display import SVG, display
import PIL
from PIL import Image
#from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)


# In[3]:


get_ipython().system(u'pip install -qU svgwrite')


# In[4]:


import svgwrite # conda install -c omnia svgwrite=1.1.6


# In[5]:


tf.logging.info("TensorFlow Version: %s", tf.__version__)


# In[6]:


#!pip install -q magenta


# In[7]:


# import our command line tools

# for testing sketch-rnn
from sketch_rnn_train import *
from model import *
'''# for testing sketch-onv2seq
from sketch_rnn_train_onv import *
from model_dnn_encoder import *
'''
'''# for testing sketch-pix2seq
from sketch_rnn_train_image import *
from model_cnn_encoder import *
'''
from utils import *
from rnn import *


# In[7]:
def draw_helper(data, factor, min_x, max_x, min_y, max_y, svg_filename, padding=50):
  diff_x = max_x - min_x
  diff_y = max_y - min_y
  size = max(diff_x, diff_y) + padding
  dims = (size, size)
  padding_x = size - diff_x
  padding_y = size - diff_y
  #print (dims, diff_x, diff_y, padding_x, padding_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = padding_x/2 - min_x 
  abs_y = padding_y/2 - min_y
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
  
def draw_strokes_sequence(data, factor, svg_prefix): 
  tf.gfile.MakeDirs(svg_prefix)
  print (data.shape)
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  for i in reversed(range(data.shape[0])):
      draw_helper(data[0:i], factor, min_x, max_x, min_y, max_y, os.path.join(svg_prefix,str(i)+'.svg'))
      
  
  

# In[8]:
# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg', padding=50):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  '''dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y'''
  diff_x = max_x - min_x
  diff_y = max_y - min_y
  size = max(diff_x, diff_y) + padding 
  dims = (size, size)
  padding_x = size - diff_x
  padding_y = size - diff_y
  #print (dims, diff_x, diff_y, padding_x, padding_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = padding_x/2 - min_x 
  abs_y = padding_y/2 - min_y
  
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
  stroke_width = 2
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  #display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)


# define the path of the model you want to load, and also the path of the dataset

# In[105]:

## change data directory for testing
data_dir = '/home/qyy/workspace/data/sketch'#add sketch for sketch-rnn model
models_root_dir = '/home/qyy/workspace/backup_models'
model_dir = os.path.join(models_root_dir, 'rnn_encoder_5classes_bs500')#'dnn_encoder_5classes_pretrainedcnn_binocular')#cat_bus_rnn_encoder_pretrained/

##cat_bus_cnn_encoder_lr0.001_bs400_64*64


# In[106]:

## change returned data
#[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model, train_images, valid_images, test_images] = load_env(data_dir, model_dir)
#[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model, 
# train_onvs_left, valid_onvs_left, test_onvs_left,  train_onvs_right, valid_onvs_right, test_onvs_right] = load_env(data_dir, model_dir)
[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)


# In[107]:


download_pretrained_models(models_root_dir=models_root_dir)


# In[108]:


# construct the sketch-rnn model here:
reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)


# In[109]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[110]:


# loads the weights from checkpoint into our model
load_checkpoint(sess, model_dir)


# We define two convenience functions to encode a stroke into a latent vector, and decode from latent vector to stroke.

# In[111]:


def encode(input_strokes):
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_param = json.load(f)
    max_seq_len = int(model_param['max_seq_len'])
    print max_seq_len
  strokes = to_big_strokes(input_strokes, max_seq_len).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]


# In[112]:


def encode_image(image):
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_param = json.load(f)
  return sess.run(eval_model.batch_z, feed_dict={eval_model.img_data: [image]})[0]


# In[113]:

def encode_onv(onv):
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_param = json.load(f)  
  return sess.run(eval_model.batch_z, feed_dict={eval_model.onv_data: [onv] })[0]
  
def encode_binocular_onv(onv_left, onv_right):
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_param = json.load(f)  
  return sess.run(eval_model.batch_z, feed_dict={eval_model.onv_data_left: [onv_left],eval_model.onv_data_right: [onv_right] })[0]
  
# In[114]:
def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.05, filename='test.svg'):
  z = None
  if z_input is not None:
    z = [z_input]
  #print(type(sample), sample)
  sample_strokes, m = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
  
  strokes = to_normal_strokes(sample_strokes)
  if draw_mode:
    draw_strokes(strokes, factor, os.path.join('/home/qyy/workspace/test/rnn_encoder_5classes_0.2',filename))
  return strokes


# Let's try to encode the sample stroke into latent vector $z$

# In[125]: 

# test single image (1) (2)
# (1) testing for sketch-rnn sequence
'''stroke = test_set.random_sample()
draw_strokes(stroke)
print (stroke.shape)
z = encode(stroke)
'''

# (2) testing for sketch-pix2seq sequence
'''sample_image = np.copy(random.choice(test_images))
display(Image.fromarray(sample_image))
sample_image = np.resize(sample_image,(64,64,1))
z = encode_image(sample_image)
'''

from onv_process import show_onv

indices = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
test_indices =list(indices) + list(indices+2500) + list(indices+5000) + list(indices+7500)+ list(indices+10000)
print (test_indices)
# test batch images (1) (2)
for index in test_indices:
    
    draw_strokes(test_set.strokes[index], 0.02, '/home/qyy/workspace/test/original/'+str(index)+'.svg')
    
    # (1) testing for sketch-onv2seq
    '''sample_onv_left = np.copy(test_onvs_left[index])
    sample_onv_right = np.copy(test_onvs_right[index])
    show_onv(sample_onv_left)
    show_onv(sample_onv_right, '/home/qyy/workspace/display_image/onv_right/'+str(index)+'.png')
    z = encode_binocular_onv(sample_onv_left, sample_onv_right)'''
    
    # (2) testing for sketch-pix2seq
    '''sample_image = np.copy(test_images[index])
    display(Image.fromarray(sample_image))
    sample_image = np.resize(sample_image,(64,64,1))
    z = encode_image(sample_image)'''
    
    # (3) testing for sketch-rnn
    #stroke = test_set.strokes[index]
    #z = encode(stroke)
    
   
    decoded_stroke = decode(z, temperature=0.2, filename=str(index)+'.svg') 
    
     # draw image sequences
    #draw_strokes_sequence(decoded_stroke, factor=0.02, svg_prefix='/home/qyy/workspace/test/image_sequence_0.02/'+str(index))
  
   
# In[390]:
'''import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

display_image_root = '/home/qyy/workspace/display_image'

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def make_array():
    from PIL import Image
    return np.array([np.asarray(Image.open('face.png').convert('RGB'))]*12)

image_list=[]    
indices = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
test_indices =list(indices) + list(indices+2500) + list(indices+5000) + list(indices+7500)+ list(indices+10000)

for dirname in os.listdir(display_image_root):
    for index in test_indices:
        fname = os.path.join(display_image_root, dirname, str(index)+'.png')
        image_arr = np.array(Image.open(fname))
        image_list.append(image_arr)
    

result = gallery(np.array(image_list), ncols=len(test_indices))
plt.imshow(result)
plt.show()'''

# In[126]:  


_ = decode(z, temperature=0.2) # convert z back to drawing at temperature of 0.8


# Create generated grid at various temperatures from 0.1 to 1.0

# In[127]:


stroke_list = []
for i in range(10):
  stroke_list.append([decode(z, draw_mode=False, temperature=0.1*i+0.1), [0, i]])
stroke_grid = make_grid_svg(stroke_list)
draw_strokes(stroke_grid)


# Latent Space Interpolation Example between $z_0$ and $z_1$

# In[39]:


# get a sample drawing from the test set, and render it to .svg
z0 = z
_ = decode(z0)


# Now we interpolate between sheep $z_0$ and sheep $z_1$

# In[40]:


#stroke = test_set.random_sample()

another_image = np.copy(random.choice(test_images))
display(Image.fromarray(another_image))
print (another_image.shape)
another_image.resize((64,64,1))

# In[41]:
z1 = encode_image(another_image)
_ = decode(z1)

# In[387]:


z_list = [] # interpolate spherically between z0 and z1
N = 10
for t in np.linspace(0, 1, N):
  z_list.append(slerp(z0, z1, t))


# In[388]:


# for every latent vector in z_list, sample a vector image
reconstructions = []
for i in range(N):
  reconstructions.append([decode(z_list[i], draw_mode=False), [0, i]])


# In[389]:


stroke_grid = make_grid_svg(reconstructions)
draw_strokes(stroke_grid)


# Let's load the Flamingo Model, and try Unconditional (Decoder-Only) Generation



