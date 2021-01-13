import numpy as np
import tensorflow as tf

from layers.subpixel import SubPixel1D
from utils import _make_normalizer, _apply_normalizer

from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU

# ----------------------------------------------------------------------------
# set default options
default_opt   = { 'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
                  'layers': 2, 'batch_size': 128 }

DRATE = 2
class AudioTfilm(tf.keras.Model):

  def __init__(self, from_ckpt=False, n_dim=None, r=2, pool_size = 4, 
              strides=4, opt_params=default_opt, log_prefix='./run'):
              super(AudioTfilm, self).__init__()
              # perform the usual initialization
              self.r = r
              self.pool_size = pool_size
              self.strides = strides
              self.n_dim=n_dim,
              self.r=r,
              self.opt_params=opt_params

  def call(self, inputs):
    print ('building model...')
    #x = self.create_model(self.n_dim, self.r)
    x = self.generator(inputs)
    return x
  
  # define generator
  def generator(self, X_train):
    X = X_train
    print(type(X_train))
    L = self.layers
    n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
    n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
    downsampling_l = []

    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
      with tf.name_scope('downsc_conv%d' % l):
        x = (Conv1D(filters=nf, kernel_size=fs, dilation_rate=DRATE,
            activation=None, padding='same', init='orthogonal'))(x)
        x = (MaxPool1D(pool_size=2,padding='valid'))(x)
        x = LeakyReLU(0.2)(x)

        # create and apply the normalizer
        nb = 128 / (2**l)

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
'''
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
'''