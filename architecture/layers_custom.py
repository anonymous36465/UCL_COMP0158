import tensorflow as tf
import numpy as np
from keras.layers import Layer
from keras import backend as K

class Diagonal(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        W_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        self.units = units
        self.activation = activation
        self.activation_fn = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        super(Diagonal, self).__init__(**kwargs)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        self.n_inputs_per_node = input_dimension / self.units

        rows = np.arange(input_dimension)
        cols = np.arange(self.units)
        cols = np.repeat(cols, self.n_inputs_per_node)
        self.nonzero_ind = np.column_stack((rows, cols))

        # print 'self.nonzero_ind', self.nonzero_ind
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dimension,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_features = x.shape[1]

        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = K.reshape(mult, (-1, int(self.n_inputs_per_node)))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, int(self.units)))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):

        config = {
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }
        base_config = super(Diagonal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SparseTF(Layer):
    """
    This layer is a sparse layer. It is used to implement the attention mechanism.

    """

    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer=None, W_regularizer=None, activation=None, 
                 use_bias=None, bias_initializer=None, bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None, **kwargs):
        self.units = units
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        super(SparseTF, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        This function is called when the layer is connected to some inputs. It is used to create the weights of the layer
        """
        input_dim = input_shape[1]

        if self.map is not None:
            self.map = self.map.astype(np.float32)

        # can be initialized directly from (map) or using a loaded nonzero_ind (useful for cloning models or create from config)
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.map)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)

        nonzero_count = self.nonzero_ind.shape[0]

        self.kernel_vector = self.add_weight(
            name="kernel_vector",
            shape=(nonzero_count,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        super(SparseTF, self).build(
            input_shape
        )  # Be sure to call this at the end. Super is used to call the parent class methods
        # self.trainable_weights = [self.kernel_vector]

    def call(self, inputs):
        tt = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        output = K.dot(inputs, tt)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "nonzero_ind": np.array(self.nonzero_ind),
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "W_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint" : tf.keras.constraints.serialize(self.kernel_constraint),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "bias_constraint" : tf.keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(SparseTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)