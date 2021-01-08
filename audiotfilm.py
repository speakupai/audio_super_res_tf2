import numpy as np
import tensorflow as tf

from scipy import interpolate
from model import Model, default_opt

from layers.subpixel import SubPixel1D, SubPixel1D_v2

import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, AveragePooling1D
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import LSTM, BatchNormalization, LeakyReLU

from tensorflow.keras.initializers import RandomNormal, Orthogonal

# ----------------------------------------------------------------------------
DRATE = 2
class AudioTfilm(Model):

  def __init__(self, from_ckpt=False, n_dim=None, r=2, pool_size = 4, 
              strides=4, opt_params=default_opt, log_prefix='./run'):
    # perform the usual initialization
    self.r = r
    self.pool_size = pool_size
    self.strides = strides
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params)

  def create_model(self, n_dim, r):
    # load inputs
    X, _, _ = self.inputs

    with tf.name_scope('generator'):
      x = X
      L = self.layers
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      #n_blocks = [ 128, 64, 32, 16, 8]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print ('building model...')

      def _make_normalizer(x_in, n_filters, n_block):
        """applies an lstm layer on top of x_in"""        
        #x_shape = tf.shape(x_in)
        #n_steps = x_shape[1] / n_block # will be 32 at training
        
        x_in_down = (MaxPool1D(pool_length=n_block, border_mode='valid'))(x_in)
         
        # pooling to reduce dimension 
        #x_shape = tf.shape(x_in_down)
        
        x_rnn = LSTM(units = n_filters, return_sequences = True)(x_in_down)
        
        # output: (-1, n_steps, n_filters)
        return x_rnn

      def _apply_normalizer(x_in, x_norm, n_filters, n_block):
        x_shape = tf.shape(x_in)
        n_steps = x_shape[1] / n_block # will be 32 at training

        # reshape input into blocks
        x_in = tf.reshape(x_in, shape=(-1, n_steps, n_block, n_filters))
        x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))
        
        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = tf.reshape(x_out, shape=x_shape)

        return x_out


      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          x = (Conv1D(filters=nf, kernel_size=fs, dilation_rate=DRATE,
              activation=None, padding='same', init='orthogonal'))(x)
          x = (MaxPool1D(pool_size=2,padding='valid'))(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**l)
        
          '''params_before = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) 
          params_after = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])'''

          x_norm = _make_normalizer(x, nf, nb)
          x = _apply_normalizer(x, x_norm, nf, nb)

          print ('D-Block: ', x.get_shape())
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate=DRATE,
                  activation=None, padding='same', init='orthogonal'))(x)
          x = (MaxPool1D(pool_size=2,padding='valid'))(x)
          x = Dropout(0.5)(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**L)
          x_norm = _make_normalizer(x, n_filters[-1], nb)
          x = _apply_normalizer(x, x_norm, n_filters[-1], nb)

      # upsampling layers
      for l, nf, fs, l_in in (zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = (Conv1D(filters=2*nf, kernel_size=fs, dilation_rate=DRATE,
                  activation=None, padding='same', init='orthogonal'))(x)
        
          x = Dropout(0.5)(x)
          x = Activation('relu')(x)
          # (-1, n, f)
          x = SubPixel1D(x, r=2) 
 
          # create and apply the normalizer
          x_norm = _make_normalizer(x, nf, nb)
          x = _apply_normalizer(x, x_norm, nf, nb)
          # (-1, n, 2f)
          x = Concatenate(axis=-1)([x, l_in])
          print ('U-Block: ', x.get_shape())
      
      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=2, kernel_size=9, 
                activation=None, padding='same', init='RandomNormal')(x)    
        x = SubPixel1D(x, r=2) 

      g = Add()([x, X])
    return g

  def predict(self, X):
    assert len(X) == 1
    x_sp = spline_up(X, self.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = self.load_batch((X,X), train=False)
    return (self.predictions, feed_dict)

# ----------------------------------------------------------------------------
# helpers
def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
