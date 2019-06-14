import tensorflow as tf

# If using Google Colab, make sure the runtime type is 'GPU'
def verify_resources():
    """Checks to make sure we have the correct Tensorflow version and that we 
    have access to a GPU
    """
    assert tf.test.gpu_device_name(), "No GPU device available. Please install GPU version of TF"
    assert '2.0' in tf.__version__, "Please use TF 2.0"
    
def load_img(img_path):
    """Loads image from file and performs some preprocessing
    Args:
        img_path (str): the path for the image file
    Returns:
        img ((1, h, w, c) tf.EagerTensor): the image as a 1 x h x w x c EagerTensor
    """
    max_dim = 512
    img = tf.io.read_file(img_path)
    # Converts compressed image to integers [0-255]
    img = tf.image.decode_image(img, channels=3)
    # Converts image from [0-255] to [0-1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Gets H and W dimensions
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    # Gets a shape proportional to the old shape such that new_long_dim == max_dim
    new_shape = tf.cast(shape*scale, tf.int32)
    # Resizes images according to: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/image_ops_impl.py#L964
    # Upsamples or downsamples as appropriate
    img = tf.image.resize(img, new_shape)
    # Unsqueezes a zeroth dimension
    img = img[tf.newaxis, :]
    return img
