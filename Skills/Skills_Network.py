# Network skills

TOWER_NAME = 'tower'

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer()
	variable = tf.Variable(initializer(shape=shape), name=name)
	return variable

def create_bias_variable(name, shape):
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return tf.Variable(initializer(shape=shape), name)

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measure the sparsity of activations.

	Args:
		x: Tensor
	Returns:
		nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

### cifar10 network
'''
	(conv + pool + norm) * 2 + fc * 3 -> 10 classes
'''
def cifar10_network(images):
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = create_variable('weights', [5, 5, 3, 64])
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = create_bias_variable('biases', [64])
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv1)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
							padding='SAME', name='pool1')
	# norm1
	norm1 = tf.contrib.layers.batch_norm(pool1, center=True, scale=True, 
							is_training=True, scope = 'norm1')

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = create_variable('weights', [5, 5, 64, 64])
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = create_bias_variable('biases', [64])
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv2)

	# norm2
	norm2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, 
										is_training=True, scope = 'norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
							strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		dim = 1
		for d in pool2.get_shape()[1:].as_list():
			dim *= d
		reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

		weights = create_variable('weights', [dim, 384])
		biases = create_bias_variable('biases', [384])
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = create_variable('weights', [384, 192])
		biases = create_bias_variable('biases', [192])
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = create_variable('weights', [192, NUM_CLASSES])
		biases = create_bias_variable('biases', [NUM_CLASSES])
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear

