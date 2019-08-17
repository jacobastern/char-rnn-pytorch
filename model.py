import tensorflow as tf

def vgg_layers(layer_names):
    """Creates a vgg model that returns a list of intermediate activations
    rather than the final output value.
    Args:
        layer_names (list of str): the layers to report the activations of
    Returns:
        model (tf.keras.Model): a Model that takes as input an image and returns
            as output the activations of the specified layers of a pretrained 
            VGG network
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # Make the network un-trainable, since we are using it just to obtain a representation
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Constructs a gram matrix for each style layer's activations. We compare
    this matrix when matching style (rather than comparing the raw activations)
    Args:
        input_tensor ((1 x h x w x c) tf.EagerTensor): one style layer
    Returns:
        gram ((1 x h x w x c) tf.EagerTensor): the gram matrix for this layer, 
            of the same shape as input_tensor
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) # result[b, c, d] = sum_i sum_j input_tensor[b, i, j, c] * input_tensor[b, i, j, d]
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) # = h * w (of the activations, not the original image)
    # Scale the gram matrix by the number of entries in the activations of this layer
    gram = result/(num_locations)
    return gram

class StyleContentModel(tf.keras.Model):
    """A wrapper for a VGG19 neural network. Returns a dictionary of activations
    rather than the final VGG19 output"""
    def __init__(self, style_layers, content_layers):
        """Initializes the StyleContentModel class by defining a VGG model that
        reports the activations of specified layers.
        Args:
            style_layers (list of str): a list of the str with the names of the 
                desired style layers to match
            content_layers (list of str): a list of the str with the names of the 
                desired content layers to match 
        """
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        # We just use the VGG network to get representations; the vgg network
        # is never trained
        self.vgg.trainable = False
        self.style_layer_len = len(style_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        
    def __call__(self, x):
        """The feed-forward method of the StyleContentModel
        Args:
            x ((1 x h x w x c) tf.EagerTensor): with values in [0, 1] - the image to optimize.
        Returns:
            (dict of dicts of tf.EagerTensor's): a dictionary containing two
                dictionaries: one holding representations of the style
                activations and the other dictionary holding the content activations
        """
        # Preprocess input image
        x = x * 255
        x_preprocessed = tf.keras.applications.vgg19.preprocess_input(x)
        
        # Get activations of vgg network for image
        activations = self.vgg(x_preprocessed)
        style_activations = activations[:self.style_layer_len]
        content_activations = activations[self.style_layer_len:]
        style_grams = [gram_matrix(style_activation) 
                      for style_activation in style_activations]
        
        style_layer_dict = {style_name:value
                         for style_name, value
                         in zip(self.style_layers, style_grams)}
        content_layer_dict = {content_name:value
                           for content_name, value
                           in zip(self.content_layers, content_activations)}
        
        return {'style':style_layer_dict, 'content':content_layer_dict}
