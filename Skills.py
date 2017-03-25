# Some skills


### CIFAR-10 maybe_download_and_extract
def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

### Make a clear folder
def make_a_clear_folder(train_dir):
  maybe_download_and_extract()
  if tf.gfile.Exises(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)

### Use tensorboard
def tensorboard_details():
  tensorboard --logdir=/path to log
  http://localhost:6006/
  图的意思：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md

### Tensorflow Saver Webpage
def tensorflow_saver_webpage():
  #webpage
  https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops/saving_and_restoring_variables

### Optimizer
def optimizer():
  def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)

  def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)

  def create_rmsprop_optimizer(learning_rate, momentum):
      return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                       momentum=momentum,
                                       epsilon=1e-5)

  optimizer_factory = {'adam': create_adam_optimizer,
                       'sgd': create_sgd_optimizer,
                       'rmsprop': create_rmsprop_optimizer}

  optimizer = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
  trainable = tf.trainable_variables()
  optim = optimizer.minimize(loss, var_list=trainable)

### Save
def save(saver, sess, logdir, step):
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
  print('Storing checkpoint to {} ...'.format(logdir), end="")
  sys.stdout.flush()

  if not os.path.exists(logdir):
      os.makedirs(logdir)

  saver.save(sess, checkpoint_path, global_step=step)
  print(' Done.')

### Load
def load(saver, sess, logdir):
  print("Trying to restore saved checkpoints from {} ...".format(logdir),
        end="")

  ckpt = tf.train.get_checkpoint_state(logdir)
  if ckpt:
      print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
      global_step = int(ckpt.model_checkpoint_path
                        .split('/')[-1]
                        .split('-')[-1])
      print("  Global step was: {}".format(global_step))
      print("  Restoring...", end="")
      saver.restore(sess, ckpt.model_checkpoint_path)
      print(" Done.")
      return global_step
  else:
      print(" No checkpoint found.")
      return None

### Train formula
def train_formula():
  loss = net.loss()
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
  saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

  try:
    saved_global_step = load(saver, sess, restore_from)
    if is_overwritten_training or saved_global_step is None:
        # The first training step will be saved_global_step + 1,
        # therefore we put -1 here for new or overwritten trainings.
        saved_global_step = -1

  except:
    print("Something went wrong while restoring checkpoint. "
          "We will terminate training to avoid accidentally overwriting "
          "the previous model.")
    raise

  ### Loop
  start_time = time.time()
  if args.store_metadata and step % 50 == 0:
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

  if step % args.checkpoint_every == 0:
    save(saver, sess, logdir, step)

### Initialization
def Initialization():
  #Initialization
  tf.contrib.layers.xavier_initializer_conv2d()
  variable = tf.Variable(initializer(shape=shape), name=name)

  #Initialize zero
  initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
  variable = tf.Variable(initializer(shape=shape), name)

### Some tricks
def some_tricks():
  #Batch normalization
  tf.contrib.layers.batch_norm(pool1, center=True, scale=True, \
                                      is_training=True, scope = 'norm1')
  #Number the name of variables
  with tf.variable_scope('conv1') as scope: