# Some skills

### Some tricks
def some_tricks():
	# Initialization
	tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)

	# Initialize zero
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	variable = tf.Variable(initializer(shape=shape), name)

	# Batch normalization
	tf.contrib.layers.batch_norm(pool1, center=True, scale=True, 
								is_training=True, scope = 'norm1')
	# Number the name of variables
	with tf.variable_scope('conv1') as scope

	# tf.add_n
	tf.add_n(inputs, name=None)
	'''
	Adds all input tensors element-wise.
	inputs: A list of Tensor objects, each with same shape and type.
	Returns: A Tensor of same shape and type as the elements of inputs.
	'''