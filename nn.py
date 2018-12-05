from __future__ import absolute_import, division, print_function

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import conv_relu_layer_bn as conv_relu_bn
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu

def conv_net(input_batch, name):
	with tf.variable_scope(name):
		#conv1: 2*2@4/2
		conv1 = conv_relu('conv1', input_batch,
                            kernel_size=2, stride=2, output_dim=4)
		print("conv1: ", conv1)
		#conv2: 2*2@4/1
		conv2 = conv_relu('conv2', conv1,
                            kernel_size=2, stride=1, output_dim=4)
		print("conv2: ", conv2)
		#conv3: 2*2@8/2
		conv3 = conv_relu('conv3', conv2,
                            kernel_size=2, stride=2, output_dim=8)
		print("conv3: ", conv3)
		#conv4: 2*2@8/1
		conv4 = conv_relu('conv4', conv3,
                            kernel_size=2, stride=1, output_dim=8)
		print("conv4: ", conv4)
		#conv5: 2*2@8/2
		conv5 = conv_relu('conv5', conv4,
                            kernel_size=2, stride=2, output_dim=8)
		print("conv5: ", conv5)
		#conv6: 2*2@8/1 tanh
		conv6 = conv('conv6', conv5,
                            kernel_size=2, stride=1, output_dim=8)
		print("conv6: ", conv6)
		tanh = tf.nn.tanh(conv6)

		return tanh

def conv_net_shallow(input_batch, name):
	with tf.variable_scope(name):
		#conv1: 2*2@4/2
		conv1 = conv_relu('conv1', input_batch,
                            kernel_size=2, stride=2, output_dim=4)
		print("conv1: ", conv1)
		#conv2: 2*2@4/1
		conv2 = conv_relu('conv2', conv1,
                            kernel_size=2, stride=1, output_dim=8)
		print("conv2: ", conv2)
		#conv3: 2*2@8/2
		conv3 = conv_relu('conv3', conv2,
                            kernel_size=2, stride=2, output_dim=16)
		print("conv3: ", conv3)
		#conv4: 2*2@8/1
		conv4 = conv_relu('conv4', conv3,
                            kernel_size=2, stride=1, output_dim=16)
		print("conv4: ", conv4)
		#conv5: 2*2@8/2
		
		
		tanh = tf.nn.tanh(conv4)

		return tanh

def conv_net_bn(input_batch, name, phase):
	with tf.variable_scope(name):
		#conv1: 2*2@4/2
		conv1 = conv_relu_bn('conv1', input_batch, phase,
                            kernel_size=2, stride=2, output_dim=4)
		print("conv1: ", conv1)
		#conv2: 2*2@4/1
		conv2 = conv_relu_bn('conv2', conv1, phase,
                            kernel_size=2, stride=1, output_dim=4)
		print("conv2: ", conv2)
		#conv3: 2*2@8/2
		conv3 = conv_relu_bn('conv3', conv2, phase,
                            kernel_size=2, stride=2, output_dim=8)
		print("conv3: ", conv3)
		#conv4: 2*2@8/1
		conv4 = conv_relu_bn('conv4', conv3, phase,
                            kernel_size=2, stride=1, output_dim=8)
		print("conv4: ", conv4)
		#conv5: 2*2@8/2
		conv5 = conv_relu_bn('conv5', conv4, phase,
                            kernel_size=2, stride=2, output_dim=8)
		print("conv5: ", conv5)
		#conv6: 2*2@8/1 tanh
		conv6 = conv('conv6', conv5, kernel_size=2, stride=1, output_dim=8)
		conv6 = tf.contrib.layers.batch_norm(conv6, center=True, scale=True, is_training=phase, scope='bn') 
		print("conv6: ", conv6)
		tanh = tf.nn.tanh(conv6)

		return tanh

def my_fc_layer(input_batch, name, output_dim, apply_dropout=False):
	with tf.variable_scope(name):
		print("input_batch: ", input_batch)
		fc7 = fc('fc', input_batch, output_dim=output_dim)
		print("fc7: ", fc7)
        if apply_dropout: fc7 = drop(fc7, 0.5)
        return fc7
