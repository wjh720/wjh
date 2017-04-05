# Some skills


import Skills_Train
import Skills_Evaluate

FLAGS.data_dir
FLAGS.train_dir
FLAGS.restore_from
FLAGS.optimizer
FLAGS.learning_rate
FLAGS.momentum

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

### CIFAR-10 maybe_download_and_extract
def maybe_download_and_extract(dest_directory = FLAGS.data_dir):
	"""Download and extract the tarball from Alex's website."""
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
					float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)

### Get arguments from terminal
def get_arguments():
	parser = argparse.ArgumentParser(description='Cifar10')
	parser.add_argument('--checkpoint', type = bool, default = True,
						help = 'Whether to train from last checkpoint.')
	return parser.parse_args()

### Make a clear folder
def make_a_clear_folder(train_dir = FLAGS.train_dir):
	maybe_download_and_extract()
	args = get_arguments()
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		if (not args.checkpoint):
			tf.gfile.DeleteRecursively(FLAGS.train_dir)
			tf.gfile.MakeDirs(FLAGS.train_dir)
	else:
		tf.gfile.MakeDirs(FLAGS.train_dir)

### Use tensorboard
def tensorboard_details():
	tensorboard --logdir=/path to log
	http://localhost:6006/
	图的意思：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md

### Optimizer
def optimizer(loss):
	def create_adam_optimizer(learning_rate, momentum):
		return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)

	def create_sgd_optimizer(learning_rate, momentum):
		return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

	def create_rmsprop_optimizer(learning_rate, momentum):
			return tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
				momentum=momentum, epsilon=1e-5)

	optimizer_factory = {'adam': create_adam_optimizer, 
						'sgd': create_sgd_optimizer, 
						'rmsprop': create_rmsprop_optimizer}

	optimizer = optimizer_factory[FLAGS.optimizer](
		learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
	trainable = tf.trainable_variables()
	optim = optimizer.minimize(loss, var_list=trainable)

	return optim

### Save
def save(saver, sess, logdir = FLAGS.train_dir, step):
	#webpage
	''' https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops/saving_and_restoring_variables '''

	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)
	print('Storing checkpoint to {} ...'.format(logdir), end="")
	sys.stdout.flush()

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	saver.save(sess, checkpoint_path, global_step=step)
	print(' Done.')

### Load
def load(saver, sess, logdir = FLAGS.restore_from):
	print("Trying to restore saved checkpoints from {} ...".format(logdir), end="")

	ckpt = tf.train.get_checkpoint_state(logdir)
	if ckpt:
		print("	Checkpoint found: {}".format(ckpt.model_checkpoint_path))
		global_step = int(ckpt.model_checkpoint_path
							.split('/')[-1]
							.split('-')[-1])
		print("	Global step was: {}".format(global_step))
		print("	Restoring...", end="")
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" Done.")
		return global_step
	else:
		print(" No checkpoint found.")
		return None

### Main
def Main():
	def main(argv=None):	# pylint: disable=unused-argument
		make_a_clear_folder()
		Skills_Train.train()
		Skills_Evaluate.evaluate()

	if __name__ == '__main__':
		main()

### Import
def Import():
	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function

	import numpy as np
	import tensorflow as tf
	from tensorflow.python.client import timeline
	import librosa

	from six.moves import xrange
	from six.moves import urllib
	
	from datetime import datetime

	import fnmatch
	import random
	import threading
	import argparse
	import tarfile
	import json
	import os.path

	import time
	import sys
	import math
	import gzip
	import os
	import re	
