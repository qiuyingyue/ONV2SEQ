# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# internal imports

import numpy as np
import tensorflow as tf

from tf_lib import HParams
import rnn, nn


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  new_hparams = HParams()
  new_hparams.update(hparams.keyvals)
  return new_hparams


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = HParams(
      data_set=['sketchrnn_cat.npz', 'sketchrnn_bus.npz','sketchrnn_dog.npz', 'sketchrnn_rabbit.npz','sketchrnn_tree.npz'],  # Our dataset.,
      num_steps=10000000,  # Total number of steps of training. Keep large.
      save_every=2000,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=256,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing. 0.01
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=500,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate. 0.001
      decay_rate=0.9999,  # Learning rate decay per minibatch. 0.9999
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate. 0.00001,
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True,  # Is model training? Recommend keeping true.
      img_size=64
  )
  return hparams


class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def cnn_encoder(self, batch):
    """Define encoder module for pixel images"""

    
    #cnn layers
    print ("batch: ", batch)
    h = tf.layers.conv2d(batch, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
    print ("h_conv1: ", h)
    h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
    print ("h_conv2: ", h)
    h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
    print ("h_conv3: ", h)
    h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
    print ("h_conv4: ", h)
    h = tf.reshape(h, [-1, 2*2*256])
    print ("h", h)

    #compute mu and pre-sigma
    mu = tf.layers.dense(h, self.hps.z_size, name="enc_fc_mu")
    presig = tf.layers.dense(h, self.hps.z_size, name="enc_fc_presig")
    print ("mu", mu)
    print("presig", presig)
    return mu, presig

  '''def cnn_encoder(self, batch, color=False, phase=True):
    """Define encoder module for pixel images"""
    
    #cnn layers
    last_h = nn.conv_net_shallow(batch, "cnn")
    print("----------")
    print("last_h: ", last_h)
    print("----------")
    #cnn layers with batch normalization
    #last_h = nn.conv_net_bn(batch, "cnn", phase)
    #l2 regularize
    last_h = tf.nn.l2_normalize(last_h, 1)

    #compute mu and pre-sigma
    mu = nn.fc_layer(last_h, "fc_mu", output_dim=self.hps.z_size)
    presig = nn.fc_layer(last_h, "fc_sigma", output_dim=self.hps.z_size)
    return mu, presig'''

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    presig = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
    return mu, presig

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = False
    if self.hps.use_recurrent_dropout == 1:
      use_recurrent_dropout = True

    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True

    if hps.dec_model == 'hyper':
      cell = cell_fn(
          hps.dec_rnn_size,
          use_recurrent_dropout=use_recurrent_dropout,
          dropout_keep_prob=self.hps.recurrent_dropout_prob)
    else:
      cell = cell_fn(
          hps.dec_rnn_size,
          use_recurrent_dropout=use_recurrent_dropout,
          dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # dropout:
    tf.logging.info('Input dropout mode = %s.', use_input_dropout)
    tf.logging.info('Output dropout mode = %s.', use_output_dropout)
    tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size])
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])

    self.input_x = self.input_data[:, :self.hps.max_seq_len, :]
    self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]

    self.img_data = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.img_size, self.hps.img_size, 1])

    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.mean, self.presig = self.cnn_encoder(self.img_data)
      self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      eps = tf.random_normal(
          (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
      self.batch_z = self.mean + tf.multiply(self.sigma, eps)
      # KL cost
      self.kl1 = -0.5 * tf.reduce_mean(self.presig)
      self.kl2 = -0.5 * tf.reduce_mean(-tf.square(self.mean))
      self.kl3 = -0.5 * tf.reduce_mean(-tf.exp(self.presig))

      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.hps.batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)

    self.num_mixture = hps.num_mixture

    # TODO(deck): Better understand this comment.
    # Number of outputs is end_of_stroke + prob + 2*(mu + sig) + corr
    n_out = (3 + self.num_mixture * 6)

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.dec_rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = last_state

    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      """Returns result of eq # 24 and 25 of http://arxiv.org/abs/1308.0850."""
      norm1 = tf.subtract(x1, mu1)
      norm2 = tf.subtract(x2, mu2)
      s1s2 = tf.multiply(s1, s2)
      z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
           2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
      neg_rho = 1 - tf.square(rho)
      result = tf.exp(tf.div(-z, 2 * neg_rho))
      denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
      """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                             z_corr)
      epsilon = 1e-6
      result1 = tf.multiply(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(result1 + epsilon)  # avoid log(0)

      fs = 1.0 - pen_data[:, 2]  # use training data for this
      fs = tf.reshape(fs, [-1, 1])
      result1 = tf.multiply(result1, fs)

      result2 = tf.nn.softmax_cross_entropy_with_logits(
          labels=pen_data, logits=z_pen_logits)
      result2 = tf.reshape(result2, [-1, 1])
      if not self.hps.is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

      result = result1 + result2
      return result

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
      """Returns the tf slices containing mdn dist params."""
      # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
      z = output
      z_pen_logits = z[:, 0:3]  # pen states
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

      # process output z's into MDN paramters

      # softmax all the pi's and pen states:
      z_pi = tf.nn.softmax(z_pi)
      z_pen = tf.nn.softmax(z_pen_logits)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)

      r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
      return r

    out = get_mixture_coef(output)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr
    self.pen = o_pen  # state of the pen
    self.pen_logits = o_pen_logits  # state of the pen

    # reshape target data so that it is compatible with prediction shape
    target = tf.reshape(self.output_x, [-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
    pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                            o_pen_logits, x1_data, x2_data, pen_data)

    self.r_cost = tf.reduce_mean(lossfunc)

    if self.hps.is_training:
      self.cost = self.r_cost + self.kl_cost * self.hps.kl_weight

      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      if self.hps.kl_weight != 0:
        self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
        self.cost = self.r_cost + self.kl_cost * self.kl_weight
      else:
        self.kl_weight = tf.Variable(self.hps.kl_weight, trainable=False)
        self.cost = self.r_cost

      gvs = optimizer.compute_gradients(self.cost)
      g = self.hps.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')


def sample(sess, model, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
  """Samples a sequence from a pre-trained model."""

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(1, model.hps.z_size)  # not used if unconditional

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []
  greedy = False
  temp = 1.0

  for i in range(seq_len):
    if not model.hps.conditional:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state
      }
    else:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.batch_z: z
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                          o_sigma1[0][idx], o_sigma2[0][idx],
                                          o_corr[0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    params = [
        o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
        o_pen[0]
    ]

    mixture_params.append(params)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_state = next_state

  return strokes, mixture_params
