# Some skills


import Skills_Network
import Skills_Input
import Skills_Loss

FLAGS.data_dir
FLAGS.train_dir
FLAGS.restore_from

FLAGS.checkpoint_every
FLAGS.max_checkpoints

### Train formula
def train(logdir = FLAGS.train_dir, restore_from = FLAGS.restore_from):
	with tf.Graph().as_default():
		# Get data, Run the model, Get the loss, Specify the optimizer.
		images, labels = Skills_Input.cifar10_inputs()
		logits = Skills_Network.cifar10_network(images)
		loss = Skills_Loss.cifar10_loss(logits, labels)
		optim = optimizer(loss)

		# Set up logging for TensorBoard.
		writer = tf.summary.FileWriter(logdir)
		writer.add_graph(tf.get_default_graph())
		run_metadata = tf.RunMetadata()
		summaries = tf.summary.merge_all()

		# Set up session
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		init = tf.global_variables_initializer()
		sess.run(init)

		# Saver for storing checkpoints of the model.
		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=FLAGS.max_checkpoints)

		# Load checkpoint
		try:
			saved_global_step = load(saver, sess, restore_from)
			if saved_global_step is None:
					# The first training step will be saved_global_step + 1,
					# therefore we put -1 here for new or overwritten trainings.
					saved_global_step = -1

		except:
			print("Something went wrong while restoring checkpoint. "
					"We will terminate training to avoid accidentally overwriting "
					"the previous model.")
			raise
		
		# Start the queue runners.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		### Loop
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			if step % 50 == 0:
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				summary, loss_value, _ = sess.run([summaries, loss, optim], options=run_options, 
													run_metadata=run_metadata)
				writer.add_summary(summary, step)
				writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
			else:
				summary, loss_value, _ = sess.run([summaries, loss, optim])
				writer.add_summary(summary, step)

			duration = time.time() - start_time
			print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

			if step % FLAGS.checkpoint_every == 0:
				save(saver, sess, logdir, step)

		#Close
		coord.request_stop()
		coord.join(threads)
