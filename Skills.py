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
tensorboard --logdir=/path to log
http://localhost:6006/

