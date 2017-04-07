# Some skills

### Some tricks for establishing the network.
def some_tricks_1():
	# Initialization
	tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)

	# Initialize zero
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	variable = tf.Variable(initializer(shape=shape), name)

	# Batch normalization
	tf.contrib.layers.batch_norm(pool1, center=True, scale=True, 
								is_training=True, scope = 'norm1')

	# Input Queue
	class tf.PaddingFIFOQueue
	__init__(capacity, dtypes, shapes, names=None, shared_name=None, name='padding_fifo_queue')
	'''
	Args:
		capacity: An integer.
			The upper bound on the number of elements that may be stored in this queue.
		dtypes: A list of DType objects.
			The length of dtypes must equal the number of tensors in each queue element.
		shapes: A list of TensorShape objects, with the same length as dtypes.
			Any dimension in the TensorShape containing value None is dynamic and 
			allows values to be enqueued with variable size in that dimension.
		names: (Optional.) A list of string naming the components in the queue with the same length as dtypes, or None. 
			If specified the dequeue methods return a dictionary with the names as keys.
		shared_name: (Optional.) If non-empty, this queue will be shared under the given name across multiple sessions.
		name: Optional name for the queue operation.
	'''
	enqueue
	enqueue_many
	dequeue
	dequeue_many


### Some tricks for writing Tensorflow.
def some_tricks_2():
	# Number the name of variables
	# 在Tensorbroad里面会合并，变量名还不会重复
	with tf.variable_scope('conv1') as scope:

	#如果我们需要定义多个Graph，则需要在with语句中调用Graph.as_default()方法将某个graph设置成默认Graph，
	#于是with语句块中调用的Operation或Tensor将会添加到该Graph中。
	with tf.Graph().as_default() as g1:

	# tf.add_n
	tf.add_n(inputs, name=None)
		'''
		Adds all input tensors element-wise.

		inputs: A list of Tensor objects, each with same shape and type.
		Returns: A Tensor of same shape and type as the elements of inputs.
		'''

	# tf.nn.in_top_k(predictions, targets, k, name=None)
		'''
		Says whether the targets are in the top K predictions.
		This outputs a batch_size bool array.

		predictions: A Tensor of type float32. A batch_size x classes tensor.
		targets: A Tensor. Must be one of the following types: int32, int64. A batch_size vector of class ids.
		k: An int. Number of top elements to look at for computing precision.
		'''

	# Wait for all the threads to terminate, give them 10s grace period
	coord.join(threads, stop_grace_period_secs=10)

	'''
	Coordinator类用来帮助多个线程协同工作，多个线程同步终止。 其主要方法有：

	should_stop():如果线程应该停止则返回True。
	request_stop(<exception>): 请求该线程停止。
	join(<list of threads>):等待被指定的线程终止。
	'''

### Some tricks for other packages.
def some_tricks_3():
	# Audio info
    sample_rate = audio_file.getframerate()
    sample_width = audio_file.getsampwidth()
    number_of_channels = audio_file.getnchannels()
    number_of_frames = audio_file.getnframes()
    '''
    	Returns sampling frequency.
		Returns sample width in bytes.
		Returns number of audio channels (1 for mono, 2 for stereo).
		Returns number of audio frames.
	'''

	# Read raw bytes
    data = audio_file.readframes(number_of_frames)
    audio_file.close()

    # __init__.py
	这个文件能让一个文件夹变成一个package，而且里面from import东西的话，代码可以直接import。

	# Stack arrays in sequence vertically (row wise).
	numpy.vstack(tup)
	'''
	tup : sequence of ndarrays.
	Tuple containing arrays to be stacked. The arrays must have the same shape along all but the first axis.

	>>> a = np.array([1, 2, 3])
	>>> b = np.array([2, 3, 4])
	>>> np.vstack((a,b))
	array([[1, 2, 3],
	       [2, 3, 4]])
	
	'''



