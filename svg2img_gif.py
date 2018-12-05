

from cairosvg import svg2png
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import os
import re
import svgwrite
import numpy as np 
from utils import *
from PIL import Image, ImageOps
from tempfile import TemporaryFile
#from scipy import misc
#from onv_process import onv_convert_fromarr

# little function that displays vector images and saves them to .svg
def save_strokes(data, factor=0.2, padding=10, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  #tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  #dims = (50 + max_x - min_x, 50 + max_y - min_y)
  #print (max_x, min_x, max_y, min_y)
  #return
  diff_x = max_x - min_x
  diff_y = max_y - min_y
  size = max(diff_x, diff_y) + padding/factor 
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
  for i in range(len(data)):
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
  stroke_width = 30
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  #display(SVG(dwg.tostring()))

def resize_strokes(strokes, size):
  new_strokes = []
  for data in strokes:
  	data = resize_single_stroke(data, size)
  	new_strokes.append(data)
  new_strokes =  np.array(new_strokes)
  print (new_strokes.shape)
  return new_strokes

def resize_single_stroke(data, size):
  min_x, max_x, min_y, max_y = get_bounds(data, 1)
  #print ("before", min_x, max_x, min_y, max_y)
  width = max_x - min_x
  height = max_y - min_y
  center_x = width / 2.0 + min_x
  center_y = height / 2.0 + min_y
  #print ("width:", width, "height", height, "center_x", center_x, "center_y", center_y)
  data[0,0] = data[0,0] - center_x
  data[0,1] = data[0,1] - center_y

  min_x, max_x, min_y, max_y = get_bounds(data, 1)
  #print ("middle", min_x, max_x, min_y, max_y)
  factor = float(size) / max(width, height) 
  #print ("factor", factor)
  data[:,0] = data[:,0] * factor
  data[:,1] = data[:,1] * factor

  min_x, max_x, min_y, max_y = get_bounds(data, 1)
  #print ("after", min_x, max_x, min_y, max_y)
  return data




def convert_with_cairosvg(svg_filename, png_filename):
  '''if os.path.exists(png_filename):
  	return'''
  svg2png(open(svg_filename, 'rb').read(), write_to=open(png_filename, 'wb'))


def strokes_to_npy(strokes, is_grey=True, size=64):
	npy = []
	print (type(strokes), strokes.shape)
	index = 0
	for data in strokes:
		#print (type(data), data.shape)
		save_strokes(data, factor=0.5, padding=10, svg_filename=svg_filename)
		convert_with_cairosvg(svg_filename, png_filename)
		im = Image.open(png_filename)
		im = im.resize((size,size), Image.ANTIALIAS)
		im2arr = np.array(im)
		if (is_grey):
			im2arr = im2arr[:,:, 1]
		npy.append(im2arr)
		'''index += 1
		if (index >= 500):
			break'''
		
	npy = np.array(npy)
	print (npy.shape)
	return npy
		
def strokes_to_png(strokes, classname, batchname, is_grey=True, size=600, padding=125, save_onv=False):
	index = 0
	npy=[]
	for data in strokes:
		save_strokes(data, factor=0.5, padding=10, svg_filename=svg_filename)
		dirname = os.path.join(data_filepath, "png_thick", classname, batchname)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		png_filename = os.path.join(dirname, str(index) + ".png")
		convert_with_cairosvg(svg_filename, png_filename)
		im = Image.open(png_filename)
		im = im.resize((size-2*padding, size-2*padding), Image.ANTIALIAS)
		padding_arr = (padding, padding, padding, padding)
		new_im = ImageOps.expand(im, padding_arr, fill=(255, 255, 255))
		new_im.save(png_filename)

		#array to onv
		if (save_onv):
			im2arr = np.array(new_im)#error
			misc.imshow(im2arr)
			onv = onv_convert_fromarr(im2arr)#eerror
			npy.append(onv)

		index+=1
		#if (index >= 100):
		#	break
		
	npy = np.array(npy)
	print (npy.shape)
	return npy

	





# svg_filepath = '/home/qyy/workspace/data/svg/sketchrnn_bus.full.npz'
# svg_data = np.load(svg_filepath, encoding="bytes")
# train_strokes = svg_data['train']
# i = 0
# for data in train_strokes:
# 	svg_filename = "sample.svg"
# 	png_filename = "sample.png"

# 	save_strokes(data, 0.2, svg_filename)
# 	convert_with_cairosvg(svg_filename, png_filename)
# 	i+=1
# 	if (i > 1):
# 		break

size = 300
padding = 125
img_ind = [0, 2500, 7520, 10010]
index = 0
img_seq =  "/home/qyy/workspace/test/image_sequence"
img_onv =  "/home/qyy/workspace/display_image/onv_left"
orig_img = "/home/qyy/workspace/test/original"

in_folder = os.path.join(orig_img, str(img_ind[0]))
in_files = [name for name in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, name))]
in_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

for ind in range(len(in_files)):
	svg_filename = os.path.join(orig_img, str(img_ind[index]), in_files[ind])
	png_folder = os.path.join(orig_img, str(img_ind[index]) + '_png')
	if not os.path.exists(png_folder):
		os.makedirs(png_folder)
	png_filename = os.path.join(orig_img, str(img_ind[index]) + '_png', in_files[ind][:-4] + '.png')
	convert_with_cairosvg(svg_filename, png_filename)
	im = Image.open(png_filename)
	new_im = im.resize((600, 600))
	new_im.save(png_filename)

# in_folder = os.path.join(img_seq, str(img_ind[index]))
# in_files = [name for name in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, name))]
# in_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

# for ind in range(len(in_files)):
# 	svg_filename = os.path.join(img_seq, str(img_ind[index]), str(ind) + '.svg')
# 	png_folder = os.path.join(img_seq, str(img_ind[index]) + '_png')
# 	if not os.path.exists(png_folder):
# 		os.makedirs(png_folder)
# 	png_filename = os.path.join(img_seq, str(img_ind[index]) + '_png', str(ind) + '.png')
# 	convert_with_cairosvg(svg_filename, png_filename)
# 	im = Image.open(png_filename)
# 	new_im = im.resize((100, 100))
# 	new_im.save(png_filename)