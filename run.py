# new script to rewrite the model
import tensorflow as tf
import h5py
import numpy as np

from in_out import load_h5
import argparse

from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU

from utils import _apply_normalizer, _make_normalizer
from layers import subpixel

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  
  train_parser.add_argument('--batch-size', type=int, default=128,
    help='training batch size')
  
  train_parser.add_argument('--logname', default='tmp-run',
    help='folder where logs will be stored')
  
  train_parser.add_argument('--layers', default=4, type=int,
    help='number of layers in each of the D and U halves of the network')
    
  train_parser.add_argument('--r', type=int, default=4,
    help='upscaling factor')
  
  train_parser.add_argument('--speaker', default='single', choices=('single', 'multi'),
    help='number of speakers being trained on')
  
  train_parser.add_argument('--pool_size', type=int, default=4,
    help='size of pooling window')
  
  train_parser.add_argument('--strides', type=int, default=4, help='pooling stide')

  return parser

# ---------------------------------------------------------------------------------------------
args = make_parser()
get_inputs = vars(args.parse_args())
print(get_inputs.keys())

# get data, the audio data in this case is stored as h5 file
X_train, Y_train = load_h5(get_inputs['train'])
X_val, Y_val = load_h5(get_inputs['val'])

# determine super-resolution level from the target audio
n_dim, n_chan = Y_train[0].shape

# get the upscaling factor 'r'
r = get_inputs['r']
assert n_chan == 1

# get variables
epochs = get_inputs['epochs']
batch_size = get_inputs['batch_size']
layers = get_inputs['layers']
speaker = get_inputs['speaker']
pool_size = get_inputs['pool_size']
strides = get_inputs['strides']
DRATE=2

def audiotfilm(X_train, Y_train, layers):
    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    L = layers
    n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
    n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
    downsampling_l = []

    x = inputs
    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
      with tf.name_scope('downsc_conv%d' % l):
        x = (Conv1D(filters=nf, kernel_size=fs, dilation_rate=DRATE,
            activation=None, padding='same', kernel_initializer='orthogonal'))(x)
        x = (MaxPool1D(pool_size=2,padding='valid'))(x)
        x = LeakyReLU(0.2)(x)

        # create and apply the normalizer
        nb = 128 / (2**l)
        x_norm = _make_normalizer(x, nf, nb)
        x = _apply_normalizer(x, x_norm, nf, nb)

        downsampling_l.append(x)

    # bottleneck layer
    with tf.name_scope('bottleneck_conv'):
        x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate=DRATE,
            activation=None, padding='same', kernel_initializer='orthogonal'))(x)
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
            activation=None, padding='same', kernel_initializer='orthogonal'))(x)
        
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        # (-1, n, f)
        x = subpixel.SubPixel1D(x, r=2) 
 
        # create and apply the normalizer
        x_norm = _make_normalizer(x, nf, nb)
        x = _apply_normalizer(x, x_norm, nf, nb)
        # (-1, n, 2f)
        x = Concatenate(axis=-1)([x, l_in])
        print ('U-Block: ', x.get_shape())
      
      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=2, kernel_size=9, activation=None, 
            padding='same', kernel_initializer='RandomNormal')(x)    
        x = subpixel.SubPixel1D(x, r=2) 

      g = Add()([x, X_train])
      return g
    outputs = tf.keras.layers.Conv1D(32, 3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.999), 
    loss='binary_crossentorpy',
     metrics='accuracy')
    
    return model

model = audiotfilm(X_train, Y_train, layers)
model.compile()
model.fit(X_train, Y_train)
model.summary()