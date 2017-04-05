# Some skills


import Skills_Network
import Skills_Input

FLAGS.data_dir
FLAGS.train_dir
FLAGS.restore_from

FLAGS.eval_dir
FLAGS.run_once
FLAGS.eval_interval_secs
FLAGS.max_checkpoints
FLAGS.checkpoint_dir

def eval_once(saver, summary_writer, top_k_op, summary_op):
	"""Run Eval once.

	Args:
		saver: Saver.
		summary_writer: Summary writer.
		top_k_op: Top K op.
		summary_op: Summary op.
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#	 /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0	# Counts the number of correct predictions.
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:	# pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

### Evaluate formula
def evaluate(logdir = FLAGS.train_dir, restore_from = FLAGS.restore_from):
	with tf.Graph().as_default():
		# Get data, Run the model, Calculate predictions,Set up logging for TensorBoard, Saver.
		eval_data = FLAGS.eval_data == 'test'
		images, labels = cifar10.inputs(eval_data=eval_data)
		logits = Skills_Network.cifar10_network(images)
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=FLAGS.max_checkpoints)

		# Set up logging for TensorBoard.
		writer = tf.summary.FileWriter(FLAGS.eval_dir)
		writer.add_graph(tf.get_default_graph())
		run_metadata = tf.RunMetadata()
		summaries = tf.summary.merge_all()

		while True:
			eval_once(saver, writer, top_k_op, summaries)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)
