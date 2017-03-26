# Network skills

### cifar10 loss
def cifar10_loss(logits, labels):
	""" Add summary for for "Loss" and "Loss/avg".
	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
						of shape [batch_size]

	Returns:
		Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels = labels, logits = logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')
