"""
This script generate the photoreceptor input for the face dataset
The log polar sampling process is converted from the C++ implementation
Please refer to "Log Polar" section for more details
"""
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/tao/miniconda2/lib/python2.7/dist-packages")
import os
import struct
import numpy as np
import gzip

import csv
from PIL import Image
import matplotlib.pyplot as plt

global R,T,x0,y0,r_min,r_max,factor, dr, dt
global coeff_00, coeff_01, coeff_10, coeff_11
global index_00,index_01,index_10,index_11




from scipy import misc

# ========================== #
#        Log Polar           #
# ========================== #

img_size = 600

x0 = 0.5+img_size/2;
y0 = 0.5+img_size/2;
r_min = 0.1
r_max = np.linalg.norm(np.array([x0,y0]))
R = 138 #radius  138
T = 360 #  360
factor = R/8 #R/8
Tfactor = 5  #1
#read noise file
ovn_size = R*T/Tfactor
csvfilename = "noise_"+str(ovn_size)+"_0.15_right.csv"     #left eye: noise_onv_0.15.csv; right eye: noise_onv_0.15_right.csv
if (os.path.exists(csvfilename)):
    print ("reading noise from", csvfilename)
    #gernerated from: np.random.normal(0, 0.1, R*T)
    with open(csvfilename) as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        noise = next(csvreader)
        noise = np.array(noise)
else:
    noise = np.random.normal(0, 0.15, R*T/Tfactor)
    with open(csvfilename, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(noise)
        print ("writing noise to", csvfilename)
    


#print ("noise:", noise.shape, np.max(noise), np.min(noise))

dr = R/factor
dt = Tfactor*2.0*np.pi/T #4*2.0*np.pi/T
r = np.arange(0, R, 1.0) #It must be float number instead of integer
t = np.arange(0, T, Tfactor*1.0)#4
phi = t * dt
rr = np.tile(r, T/Tfactor)
#print ("r_max", r_max, "rr", len(rr), "noise", len(noise), "R",R)
tau = r_max * np.exp(rr/factor + noise - R/factor)
tt = np.tile(phi, R) + noise

X = x0 + np.multiply(tau, np.cos(tt))
Y = y0 + np.multiply(tau, np.sin(tt))
#print ("X", X, "Y", Y)
X_min = np.floor(X).astype(int)
Y_min = np.floor(Y).astype(int)
X_max = X_min + 1
Y_max = Y_min + 1

U = X - X_min
V = Y - Y_min

X_min = np.clip(X_min, 0, img_size-1)
Y_min = np.clip(Y_min, 0, img_size-1)
X_max = np.clip(X_max, 0, img_size-1)
Y_max = np.clip(Y_max, 0, img_size-1)

index_00 = X_min * img_size + Y_min
index_01 = X_min * img_size + Y_max
index_10 = X_max * img_size + Y_min
index_11 = X_max * img_size + Y_max

coeff_00 = np.multiply(1-U, 1-V)
coeff_01 = np.multiply(1-U, V)
coeff_10 = np.multiply(U, 1-V)
coeff_11 = np.multiply(U, V)

#print ("index_00", index_00, "index_01", index_01, "index_10", index_10, "index_11", index_11)
'''onv = np.zeros((R*T/Tfactor))
plt.scatter(Y,-X, c=onv/255.0, cmap='gray', s=1, marker='.')
plt.show()'''

def onv_convert_fromarr(I, resize=False):
    if (resize):
        I = misc.imresize(I, (600,600))
    f00 = np.take(I, index_00)
    f01 = np.take(I, index_01)
    f10 = np.take(I, index_10)
    f11 = np.take(I, index_11)
    temp = np.multiply(coeff_00,f00) + np.multiply(coeff_01,f01) + np.multiply(coeff_10,f10) + np.multiply(coeff_11,f11)
    temp = temp.astype(np.dtype('uint8'))
    #print ("temp", type(temp), temp.shape, np.mean(temp), temp)
    #show_onv(temp)
    return temp

def show_onv(onv, filename=None):
    """onv: the onv vector for display"""
    plt.scatter(Y,-X, c=onv/255.0, cmap='gray', s=10, marker='.',facecolor='0.5', lw = 0, edgecolor='r')
    plt.ylim(-600, 0)
    plt.xlim(0, 600)
    if (filename is None):
        plt.show()
    else:
        plt.savefig(filename)
        plt.show()

# =========================== #
#  Process the train dataset  #
# =========================== #
#import h5py
#import cv2

# from pylab import *
'''def process(file_dir, save_dir):


    result = np.zeros((R*T,1))
    for f in os.listdir(file_dir):
        if (not f.endswith(".png") and not f.endswith(".jpg")):
            continue
        filename = os.path.join(file_dir, f)
        print ("filename",filename)
        I = cv2.imread(filename, 0)
        I = cv2.resize(I, dsize=(img_size, img_size))

        temp = onv_convert_fromarr(I)
        print (temp.shape)

        result = temp.reshape((R*T, 1))
        # test = filename.split('_p')[1].split('_e')[0]

        # # TODO: Modified the new label for new data
        # result[R * T, :] = int(test)
        # result = result.astype(np.dtype('uint8'))


        #plt.scatter(X,Y,c=temp/255.0)
        plt.scatter(Y,-X, c=temp/255.0, cmap='gray', s=1, marker='.')
        plt.show()

        h5f = h5py.File(os.path.join(save_dir, '%s.h5'%f), 'w')
        h5f.create_dataset('data', data=np.transpose(result), compression="gzip", compression_opts=9)
        h5f.close()
process(file_dir = '../data/testimage', save_dir = '../data/testonv')'''

def png_to_onv(file_dir, onv_filepath):

    for classname in os.listdir(file_dir):
        if ( classname == "sketchrnn_tree"):
            continue
        print ("classname", classname)

        npy=[]
        idx = 0
        for f in os.listdir(os.path.join(file_dir, classname, "train")):
        #for idx in range(100):
            png_filename = os.path.join(file_dir, classname, "train", str(idx)+".png")
            #print png_filename
            im = misc.imread(png_filename, mode='L')
            #uni,count = np.unique(im, return_counts=True)
            #print (uni, count)
            # print (im.shape)
            #misc.imshow(im)
            onv = onv_convert_fromarr(im)

            npy.append(onv)
            idx+=1
        train_onv = np.array(npy)
        
        npy=[]
        idx = 0
        for f in os.listdir(os.path.join(file_dir, classname, "valid")):
        #for idx in range(100):
            png_filename = os.path.join(file_dir, classname, "valid", str(idx)+".png")
            im = misc.imread(png_filename, mode='L')
            onv = onv_convert_fromarr(im)
            npy.append(onv)
            idx+=1
        valid_onv = np.array(npy)
        
        npy=[]
        idx = 0
        for f in os.listdir(os.path.join(file_dir, classname, "test")):
        #for idx in range(100):
            png_filename = os.path.join(file_dir, classname, "test", str(idx)+".png")
            im = misc.imread(png_filename, mode='L')
            onv = onv_convert_fromarr(im)
            npy.append(onv)
            idx+=1
        test_onv = np.array(npy)

        outfile = os.path.join(onv_filepath,  classname)
        if (not os.path.exists(onv_filepath)):
            os.makedirs(onv_filepath)
        np.savez(outfile, train=train_onv, valid=valid_onv,test=test_onv)

        # data = np.load("../data/onv/sketchrnn_rabbit.npz")
        # plt.scatter(Y,-X, c=data["valid"][0]/255.0, cmap='gray', s=1, marker='.')
        # print (np.mean(data["valid"][0]))
        # plt.show()
if __name__ == "__main__":
    png_to_onv("../data/png_thick", "../data/onv_9936_thick_right")
