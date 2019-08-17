import tensorflow as tf

def high_pass_x_y(image):
    """Detects horizontal and vertical edges in an image.
    Args:
        image ((1 x h x w x c) tf.ResourceVariable): the image we are optimizing
    Returns:
        x_var ((1 x h x (w - 1) x c) tf.EagerTensor): a matrix of deltas
            with larger values where there are horizontal edges
        y_var ((1 x (h - 1) x w x c) tf.EagerTensor): a matrix of deltas
            with larger values where there are vertical edges
    """
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def sobel(image):
    """Another alternative for an edge detector; detects horizontal and 
    vertical edges in an image
    Args:
        image ((1 x h x w x c) tf.ResourceVariable): the image we are optimizing
    Returns:
        x_var ((1 x h x (w - 1) x c) tf.EagerTensor): a matrix of deltas
            with larger values where there are horizontal edges
        y_var ((1 x (h - 1) x w x c) tf.EagerTensor): a matrix of deltas
            with larger values where there are vertical edges
    """
    sobel = tf.image.sobel_edges(content_image)
    x_var = sobel[...,0]/4+.5
    y_var = sobel[...,1]/4+.5
    return x_var, y_var

def style_content_loss(outputs):
    """A loss function that penalizes the style and content outputs for being
    different from the style and content targets.
    Args:
        outputs (dict of dict of tf.EagerTensor): the activations from the 
            StyleContentModel
    Returns:
        loss (() tf.Tensor): the style and content loss
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    # Gather MSE loss across each layer
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                           for name in style_outputs.keys()])
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                             for name in content_outputs.keys()])
    # Scale appropriately
    style_loss *= style_weight / num_style_layers
    content_loss *= content_weight / num_content_layers
    
    loss = style_loss + content_loss
    return loss

def total_variation_loss(image):
    """Obtains the regularization loss that penalizes the stylized image for 
    having high-frequency components
    Args:
        image ((1 x h x w x c) tf.EagerTensor): the stylized image
    Returns:
        (() tf.Tensor): the regularization penalty for the loss function
    """
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)    

@tf.function()
def train_step(image):
    """Conducts one step of training with 'image' and updates the values of 'image'
    Args:
        image ((1 x h x w x c) tf.EagerTensor): the image to optimize    
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*total_variation_loss(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    
def clip_0_1(image):
    """Clips image values to fall in the proper range ([0, 1]).
    Args:
        image ((1 x h x w x c) tf.ResourceVariable): the image with raw values
    Returns:
        ((1 x h x w x c) tf.Tensor): the image with values clipped to be in [0, 1]
    """
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
