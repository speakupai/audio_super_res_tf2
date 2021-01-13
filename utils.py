# create tf2 version of the model

import tensorflow as tf
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import LSTM
import numpy as np
import tensorflow.keras.backend as K
from dataset import DataSet

import os
import time

import pickle
import librosa

def _make_normalizer(x_in, n_filters, n_block):
	"""applies an lstm layer on top of x_in"""
	x_in_down = (MaxPool1D(pool_size=n_block, padding='valid'))(x_in)        
	x_rnn = LSTM(units = n_filters, return_sequences = True)(x_in_down)

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


def get_power(x):
	S = librosa.stft(x, 2048)
	S = np.log(np.abs(S)**2 + 1e-8)

	return S

def compute_log_distortion(x_hr, x_pr):
	S1 = get_power(x_hr)
	S2 = get_power(x_pr)
	lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis = 0)
	return min(lsd, 10.)

def loss(Y, y_hat):
	# load model output and true output
	P = y_hat

	# compute l2 loss
	sqrt_l2_loss = tf.math.sqrt(tf.math.reduce_mean((P-Y)**2 + 1e-6, axis=[1,2]))
	avg_sqrt_l2_loss = tf.math.reduce_mean(sqrt_l2_loss, axis=0)

	return avg_sqrt_l2_loss

def calc_snr(Y, Pred):
	sqrt_l2_loss = np.sqrt(np.mean((Pred-Y)**2+1e-6, axis=(0, 1)))
	sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(0,1)))
	snr = 20* np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
	return snr

def calc_snr2(Y, P):
	sqrt_l2_loss = np.sqrt(np.mean((P-Y)**2 + 1e-6, axis=(1,2)))
	sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(1,2)))
	snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
	avg_snr = np.mean(snr, axis=0)
	return avg_snr