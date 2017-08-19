# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# this VAE implementation is based on Jan Hendrik Metzen's code
# https://jmetzen.github.io/2015-11-27/vae.html

import numpy as np
import tensorflow as tf
import tf_util as U

def xavier_init(fan_in, fan_out, constant=1): 
  """ Xavier initialization of network weights"""
  # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  low  = -constant * np.sqrt(6.0/(fan_in + fan_out)) 
  high =  constant * np.sqrt(6.0/(fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out), 
                           minval=low,
                           maxval=high, 
                           dtype=tf.float32)


class LatentRL(object):
  """ Variation Autoencoder """
  def __init__(self,
               ob_space,
               ac_space,
               num_control,
               learning_rate,
               beta):

    self._ob_space  = ob_space
    self._ac_space  = ac_space
    self.pdtype     = make_pdtype(ac_space)

    self._create_network()
    self._prepare_loss(learning_rate, beta)

  def _create_network(self):

    n_hidden_encode   = [32, 32]
    n_hidden_v        = [16, 16]
    n_hidden_mlp      = [32]
    n_input           = self._ob_space.shape
    n_z               = 10
    n_w               = 4         # Need to test with value, 
    n_v               = n_z - n_w # Conditionally independent variables, from beta-VAE
    n_t               = 5        # task specific output size


    # n_hidden_encode_1 = 32
    # n_hidden_encode_2 = 32
    # n_hidden_mlp_1 = 1200
    # n_hidden_mlp_2 = 1200
    # n_hidden_decode_3 = 1200
    # n_input           = self._ob_space.shape
    # n_z               = 10
    # n_v               = 4 # Need to test with value, Conditionally independent variables, from beta-VAE
    # n_w               = n_z - n_v
    
    # network_weights = self._initialize_weights(n_hidden_encode_1,
    #                                            n_hidden_encode_2,
    #                                            n_hidden_mlp_1,
    #                                            n_hidden_mlp_2,
    #                                            n_input,
    #                                            n_z,
    #                                            n_v)

    sequence_length = None
    self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
    self.c_in = U.get_placeholder(name"c_in", dtype = tf.int32, shape=[sequence_length, num_control])
    # need something related to observation normalization

    self.obz = self.ob
    
    self.z_mean, self.z_log_sigma_sq = \
      self._create_encoder_network(hidden_layers = n_hidden_encode, out_sz = n_z)

    # batch_sz = tf.shape(self.x)[0]
    batch_sz = tf.shape(self.ob)[0]
    eps_shape = tf.stack([batch_sz, n_z])

    # mean=0.0, stddev=1.0    
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    
    # z = mu + sigma * epsilon
    self.z   = tf.add(self.z_mean, 
                    tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
    self.z_w = self.z[:, :n_w]
    self.z_v = self.z[:, n_w:]

    self.v_task = self._create_task_network(hidden_layers=n_hidden_v, out_sz = n_t)
    self.pol_in = tf.concat([self.z_w, self.v_task], axis=1)

    self.vpred  = 
    self.



    self._create_decoder_network(network_weights["weights_decode"],
                                   network_weights["biases_decode"])
    

    self.x_reconstr_mean_logit, self.x_reconstr_mean = \
      self._create_decoder_network(network_weights["weights_decode"],
                                   network_weights["biases_decode"])

  
  def _create_encoder_network(self, hidden_layers, latent_sz):
    last_out = self.obz
    for (i, hid_size) in enumerate(hidden_layers):
        last_out    = tf.nn.relu(U.dense(last_out, hid_size, "enc%i"%(i+1), weight_init=tf.contrib.layers.xavier_initializer()))

    # self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

    z_mean          = U.dense(last_out, latent_sz, "enc_mean", weight_init=tf.contrib.layers.xavier_initializer())
    z_log_sigma_sq  = U.dense(last_out, latent_sz, "enc_sigma", weight_init=tf.contrib.layers.xavier_initializer())


    # layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x,  weights['h1']), biases['b1']))
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # z_mean         = tf.add(tf.matmul(layer_2, weights['out_mean']),
    #                         biases['out_mean'])
    # z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
    #                         biases['out_log_sigma'])
    return (z_mean, z_log_sigma_sq)

  def _create_task_network(self, hidden_layers, out_sz):
    last_out = tf.concat([self.z_w, self.c_in], axis = 1)

    for (i, hid_size) in enumerate(hidden_layers):
        last_out    = tf.nn.tanh(U.dense(last_out, hid_size, "task%i"%(i+1), weight_init=U.normc_initializer(1.0)))

    out = tf.nn.tanh(U.dense(last_out, out_sz, "task_final", weight_init=U.normc_initializer(1.0)))  

    return out 

  def _create_core_mlp_network(self, hidden_layers, out_sz):
    last_out = self.z_w
    for (i, hid_size) in enumerate(hidden_layers):
        last_out    = tf.nn.tanh(U.dense(last_out, hid_size, "core_mlp%i"%(i+1), weight_init=U.normc_initializer(1.0)))

    out = tf.nn.tanh(U.dense(last_out, out_sz, "core_mlp_final", weight_init=U.normc_initializer(1.0)))  

    return out 


  def _create_decoder_network(self, weights, biases):
    layer_1 = tf.tanh(tf.add(tf.matmul(self.z,  weights['h1']), biases['b1']))
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    x_reconstr_mean_logit = tf.add(tf.matmul(layer_3, weights['out_mean']),
                                   biases['out_mean'])
    x_reconstr_mean = tf.nn.sigmoid(x_reconstr_mean_logit)
    return x_reconstr_mean_logit, x_reconstr_mean
      
  def _prepare_loss(self, learning_rate, beta):
    # reconstruction loss (the negative log probability)
    reconstr_loss = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                              logits=self.x_reconstr_mean_logit),
      1)
    
    # latent loss
    latent_loss = beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                              - tf.square(self.z_mean)
                                              - tf.exp(self.z_log_sigma_sq), 1)

    # average over batch
    self.loss = tf.reduce_mean(reconstr_loss + latent_loss)
    self.optimizer = tf.train.AdagradOptimizer(
      learning_rate=learning_rate).minimize(self.loss)
    
  def partial_fit(self, sess, X, summary_op=None):
    """Train model based on mini-batch of input data.
    Return loss of mini-batch.
    """
    if summary_op is None:
      _, loss = sess.run( (self.optimizer, self.loss), 
                          feed_dict={self.x: X} )
      return loss
    else:
      _, loss, summary_str = sess.run( (self.optimizer, self.loss, summary_op),
                                       feed_dict={self.x: X} )
      return loss, summary_str
      
  
  def transform(self, sess, X):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: X} )
  
  def generate(self, sess, z_mu=None):
    """ Generate data by sampling from latent space. """
    if z_mu is None:
      z_mu = np.random.normal(size=(1,10))
    return sess.run( self.x_reconstr_mean, 
                     feed_dict={self.z: z_mu} )
  
  def reconstruct(self, sess, X):
    """ Use VAE to reconstruct given data. """
    return sess.run( self.x_reconstr_mean, 
                     feed_dict={self.x: X} )
