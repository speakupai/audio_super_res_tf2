# new script to rewrite the model
import tensorflow as tf
import h5py
import numpy as np

from in_out import load_h5
import argparse
from audiotfilm import AudioTfilm

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  #train_parser.set_defaults(func=train)

  train_parser.add_argument('--model', default='audiotfilm',
    choices=('audiotfilm'), help='only audiotfilm used')
  
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
  
  train_parser.add_argument('--alg', default='adam',
    help='optimization algorithm')
  
  train_parser.add_argument('--lr', default=1e-3, type=float,
    help='learning rate')
  
  train_parser.add_argument('--r', type=int, default=4,
    help='upscaling factor')
  
  train_parser.add_argument('--speaker', default='single', choices=('single', 'multi'),
    help='number of speakers being trained on')
  
  train_parser.add_argument('--pool_size', type=int, default=4,
    help='size of pooling window')
  
  train_parser.add_argument('--strides', type=int, default=4, help='pooling stide')
  
  train_parser.add_argument('--full', default='false', choices=('true', 'false'))

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
r = Y_train[0].shape[1] / X_train[0].shape[1]
assert n_chan == 1

def audiotfilm():
    
model = AudioTfilm()
model.compile()
model.fit(X_train, Y_train)
model.summary()