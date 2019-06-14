# Install tensorflow 2.0 if you don't already have it
# !pip install tensorflow-gpu==2.0.0-beta0

# Local imports
from .model import StyleContentModel
from .utils import load_image, verify_resources
from .train import train_step

import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--content_path', required=False, type=str, default=None, 
                        help='The path for the content image (default: an image of a turtle)')
    parser.add_argument('-s', '--style_path', required=False, type=str, default=None, 
                        help='The path for the style image (default: a Kandinsky painting)')
    parser.add_argument('-o', '--output_path', required=False, type=str, default='/style_transfer_results/result.png', 
                        help='The path for the style-transfer image (default: a Kandinsky painting)')
    parser.add_argument('-n', '--num_opt_steps', required=False, type=np.int32, default=1000, 
                        help='The number of optimization steps (default: 1e-2)')
    parser.add_argument('-v', '--verbose', required=False, type=bool, default=False, 
                        help='Whether to print status updates (default: 1e-2)')
    parser.add_argument('--style_weight', required=False, type=np.float32, default=1e-2, 
                        help='A weight for the importance of the style image (default: 1e-2)')
    parser.add_argument('--content_weight', required=False, type=np.float32, default=1e4, 
                        help='A weight for the importance of the content image (default: 1e-2)')
    parser.add_argument('--total_variation_weight', required=False, type=np.float32, default=1e8, 
                        help='A weight for the importance of smoothing (to reduce noise) (default: 1e-2)')
    parser.add_argument('--style_layers', required=False, type=str, default=None, 
                        choices=['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 
                                 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 
                                 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 
                                 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'],
                        help='The style layers to match.')
    parser.add_argument('--content_layers', required=False, type=str, default=None, 
                        choices=['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 
                                 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 
                                 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 
                                 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'],
                        help='The content layers to match.')

    args = parser.parse_args()
    
    # Constants
    style_weight = args.style_weight
    content_weight = args.content_weight
    total_variation_weight = args.total_variation_weight
    opt_steps = args.num_opt_steps
    verbose = args.verbose
    content_path = args.content_path
    style_path = args.style_path
    output_path = args.output_path
    content_layers = args.content_layers
    style_layers = args.style_layers
    
    # Verify correct version of Tensorflow, verify access to GPU
    verify_resources()
    
    if content_path is None:
        # Downloads the file from the link, returns the path of the downloaded file
        content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
    if style_path is None:
        style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    if content_layers is None:
        # The layers of the vgg representation of the content image that we want to match
        content_layers = ['block5_conv2']
    if style_layers is None:
        # The layers of the vgg representation of the style image that we want to match
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    
    extractor = StyleContentModel(style_layers, content_layers)
    content_targets = extractor(content_image)['content']
    style_targets = extractor(style_image)['style']

    # Make the image a tf.Variable (technically, a tf.ResourceVariable as of TF 2.0) so that we can optimize over it
    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=.02, beta_1=.99, epsilon=1e-1)

    # Run the optimization
    start = time.time()
    for step in range(opt_steps):
        train_step(image)
        if verbose:
            if step % 20 == 0:
                print(f"Step {step} out of {opt_steps}.")
    end = time.time()
    if verbose:
        print(f"Total time: {end - start}")
    
    # Save the result
    if not os.path.exists():
        os.makedirs(os.path.dirname(output_path))
    mpl.image.imsave(output_path, image[0])
