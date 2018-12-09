

from cairosvg import svg2png
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import os
import svgwrite
import numpy as np 
from utils import *
from PIL import Image, ImageOps
from tempfile import TemporaryFile
#from scipy import misc
#from onv_process import onv_convert_fromarr, show_onv


def save_strokes(data, factor=0.2, padding=10, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  """little function that transfer the data(strokes) to .svg"""
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
  """return resized strokes"""
  new_strokes = []
  for data in strokes:
  	data = resize_single_stroke(data, size)
  	new_strokes.append(data)
  new_strokes =  np.array(new_strokes)
  print (new_strokes.shape)
  return new_strokes

def resize_single_stroke(data, size):
  """helper function for resize_single_stroke()"""
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
  """helper function to convert a svg file  to a png file"""
  '''if os.path.exists(png_filename):
  	return'''
  svg2png(open(svg_filename, 'rb').read(), write_to=open(png_filename, 'wb'))


def strokes_to_npy(strokes, is_grey=True, size=64):
	"""convert strokes to svg files and then to png files and then resize to size*size and return  numpy array"""
	npy = []
	print (type(strokes), strokes.shape)
	
	for data in strokes:
		#print (type(data), data.shape)
		svg_filename = "tmp.svg"
		png_filename = "tmp.png"
		save_strokes(data, factor=0.5, padding=10, svg_filename=svg_filename)
		convert_with_cairosvg(svg_filename, png_filename)
		im = Image.open(png_filename)
		im = im.resize((size,size), Image.ANTIALIAS)
		im2arr = np.array(im)
		if (is_grey):
			im2arr = im2arr[:,:, 1]
		npy.append(im2arr)
		
		
	npy = np.array(npy)
	print (npy.shape)
	return npy
		
def strokes_to_png(strokes, classname, batchname, png_filepath, size=600, padding=125, save_onv=False):
	"""convert strokes to svg files and then to png files and then resize to size*size with padding arround 
	    save the png files and return the numpy array"""
	""" set save_ong to True to save the onv files"""

	npy=[]
	index = 0
	for data in strokes:
		svg_filename = "tmp.svg"
		save_strokes(data, factor=0.5, padding=10, svg_filename=svg_filename)
		dirname = os.path.join(png_filepath, classname, batchname)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		png_filename = os.path.join(dirname, str(index) + ".png")
		convert_with_cairosvg(svg_filename, png_filename)
		im = Image.open(png_filename)
		im = im.resize((size-2*padding, size-2*padding), Image.ANTIALIAS)
		padding_arr = (padding, padding, padding, padding)
		new_im = ImageOps.expand(im, padding_arr, fill=(255, 255, 255))
		new_im.save(png_filename)
		index += 1
		#array to onv
		if (save_onv):
			im2arr = np.array(new_im)#error
			#misc.imshow(im2arr)
			onv = onv_convert_fromarr(im2arr)#eerror
			show_onv(onv)
			npy.append(onv)
		
	npy = np.array(npy)
	print (npy.shape)
	return npy

### Data preprocessing on original sketch files to obtain conrresponding png/onv/numpy image
def main1():
	data_filepath = '../data/'
	input_dirname = "sketch"
	output_dirname = "testmain1" # change 
	is_grey = True 
	for f in os.listdir(os.path.join(data_filepath, input_dirname)):
		print(f)
		if ("full" in f):
			continue
		classname = f.split('.')[0]
		outfile = os.path.join(data_filepath, output_dirname, f)
		if os.path.exists(outfile):
			continue
		if not os.path.exists(os.path.dirname(outfile)):
			os.makedirs(os.path.dirname(outfile))

		# load data
		fname = os.path.join(data_filepath, input_dirname, f)
		data = np.load(fname, encoding="bytes")
		train_strokes = data['train']
		valid_strokes = data['valid']
		test_strokes = data['test']

		### For(1)(2)(3) only one can work at a time

		# (1)resize the strokes and save to outfile
		'''resized_train_strokes = resize_strokes(train_strokes, size = 600)
		resized_valid_strokes = resize_strokes(valid_strokes, size = 600)
		resized_test_strokes = resize_strokes(test_strokes, size = 600)
		np.savez(outfile, train=resized_train_strokes, valid=resized_valid_strokes,test=resized_test_strokes)'''

		# (2)save the strokes into .png (and .onv optionally), intended for sketch-onv2seq
		png_filepath = os.path.join(data_filepath, "testpng")
		train_onv = strokes_to_png(train_strokes, classname, 'train', png_filepath=png_filepath, save_onv=False)
		valid_onv = strokes_to_png(valid_strokes, classname, 'valid', png_filepath=png_filepath, save_onv=False)
		test_onv = strokes_to_png(test_strokes, classname, 'test', png_filepath=png_filepath, save_onv=False)
		if save_onv:
			np.savez(outfile, train=train_onv, valid = valid_onv, test = test_onv)

		# (3)save the strokes into numpy array of images, intended for sketch-pix2seq
		'''train_images = strokes_to_npy(train_strokes, is_grey)
		valid_images = strokes_to_npy(valid_strokes, is_grey)
		test_images = strokes_to_npy(test_strokes, is_grey)	
		np.savez(outfile, train=train_images, valid=valid_images,test=test_images)'''
		
### convert .svg to .png from svg_filepath/dirname to png_filepath/dirname
def main2():
	svg_filepath = '../display_svg'
	png_filepath = '../display_image'
	svg_dirlist = ['original']
	for dirname in svg_dirlist:
		for f in os.listdir(os.path.join(svg_filepath, dirname)):
			if not 'svg' in f:
				continue
			svg_filename = os.path.join(svg_filepath, dirname, f)
			png_filename = os.path.join(png_filepath, dirname, f.replace('svg','png'))
			print ("svg_filename", svg_filename, "png_filename", png_filename)
			if (not os.path.exists(os.path.dirname(png_filename))):
				os.makedirs(os.path.dirname(png_filename))
			convert_with_cairosvg(svg_filename=svg_filename, png_filename=png_filename)
			im = Image.open(png_filename)
			im.save(png_filename)
		

def gallery(array, ncols=3):
	print (array.shape, array[0].shape)
	nindex, height, width, intensity = array.shape
	nrows = nindex//ncols
	assert nindex == nrows*ncols
	# want result.shape = (height*nrows, width*ncols, intensity)
	result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
	return result

###place multiple images to a grid for display
def main3():
	source_path = '../display_image'
	dirlist = ['original', 'onv_left_resize', 'rnn_encoder_5classes_0.2','cnn_encoder_5classes_0.2',
	'dnn_encoder_5classes_binocular_0.2','dnn_encoder_5classes_pretrainedrnn_binocular_0.2','dnn_encoder_5classes_pretrainedcnn_binocular_0.2']
	'''idx_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 
	2500, 2505, 2510, 2515, 2520, 2525, 2530, 2535, 2540, 2545, 2550, 2555, 2560, 2565, 2570, 2575, 2580, 2585, 2590, 2595, 
	5000, 5005, 5010, 5015, 5020, 5025, 5030, 5035, 5040, 5045, 5050, 5055, 5060, 5065, 5070, 5075, 5080, 5085, 5090, 5095, 
	7500, 7505, 7510, 7515, 7520, 7525, 7530, 7535, 7540, 7545, 7550, 7555, 7560, 7565, 7570, 7575, 7580, 7585, 7590, 7595, 
	10000, 10005, 10010, 10015, 10020, 10025, 10030, 10035, 10040, 10045, 10050, 10055, 10060, 10065, 10070, 10075, 10080, 10085, 10090, 10095]
	'''
	''' '''
	idx_list = [0, 25, 45, 50,  #75,
	2525, 2530, 2535,   2560, #2555,
	7515, 7540,7575, 7580, # 7570, 
	10005, 10010,  10045,  10060]#, 10090
	img_list = []#None

	for dirname in dirlist:
		for idx in idx_list:
			png_filename = os.path.join(source_path, dirname, str(idx)+'.png')
			img = Image.open(png_filename)
			img = img.resize((100,100), Image.ANTIALIAS)
			img_arr = np.array(img)
			img_list.append(img_arr[:,:,0:3])
	
	img_list = np.array(img_list)
	result = gallery(img_list, len(idx_list))
	display_img = Image.fromarray(result)
	display_img.show()
	display_img.save("../display_20_4.png")
	
		
if __name__ == "__main__":
	main1()
	#main2()
	#main3()