
import tensorflow as tf
import keras
import numpy as np
class Squeeze(keras.layers.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        return tf.squeeze(inputs)

# class Squeeze(keras.layers.Layer):
#     def __init__(self, squeeze_batch_dimension=True):
#         self.squeeze_batch_dimension = squeeze_batch_dimension
#         super(Squeeze, self).__init__()

#     def call(self, inputs):
#         squeezed_input = tf.squeeze(inputs)
#         if(self.squeeze_batch_dimension==False and tf.shape(inputs)[0]==1):
#             squeezed_input = tf.expand_dims(squeezed_input,0)
#         return squeezed_input
    
def get_truncated_model(model:tf.keras.Model, num_layers):
    truncated_model = tf.keras.models.clone_model(model)
    with tf.device('cpu'):
        model_inputs = keras.Input(shape=model.layers[0].input_shape[0][1:])
        x = model_inputs
        for i in range(num_layers):
            truncated_model.layers[i].set_weights(model.layers[i].get_weights()) 
            x = truncated_model.layers[i](x)
        new_model = tf.keras.Model(inputs = model_inputs, outputs = x)
    return new_model

def bounded_output(x, lower, upper):
    # https://stackoverflow.com/questions/62562463/constraining-a-neural-networks-output-to-be-within-an-arbitrary-range
    scale = upper - lower
    return scale * tf.nn.sigmoid(x) + lower

def get_torch_nn_parameters(model, p_type:str="all")->int:
    """Return number of all/trainable/non-trainable parameters in input model

    Args:
        model (Module): input torch model
        p_type (str, optional): choose between all/trainable/non-trainable parameters. Defaults to "all".

    Raises:
        ValueError: when p_type is not all/trainable/non-trainable
        
    Returns:
        int: number of all/trainable/non-trainable parameters 
    """
    if(p_type=="all"):
        p_filter = lambda p: True 
    elif(p_type=="trainable"):
        p_filter = lambda p: p.requires_grad
    elif(p_type=="non-trainable"):
        p_filter = lambda p: not(p.requires_grad)
    else:
        raise ValueError("Argument p filter shold be all/trainable/non-trainable")
    model_parameters = filter(p_filter,model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params
    
if __name__ == '__main__':
    pass