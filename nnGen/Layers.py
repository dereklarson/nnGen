""" This module contains all of the layer classes:
    InputLayer
    OutputLayer
    FCLayer
    ConvLayer
    PoolLayer
"""


import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared
Tsig = T.nnet.sigmoid


class Layer(object):

    def __init__(self, input_layer, traits, _tag):
        self.tag = _tag
        self.number = traits['number']
        self.input_layer = input_layer


class InputLayer(Layer):

    """ This layer will come first in any structure definition of a network.
    It is involved in applying data augmentation, such as dropout, jitter
    and flipping.
    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation=None):
        super(InputLayer, self).__init__(input_layer, traits, "Input")
        self.srng = rngs[1]
        self.out_shape = Lshape
        self.p_retain = (1. - traits['dropout'])
        self.traits = traits

    def output(self, use_dropout=False, depth=0):
        """ Provides data to next layer and applies dropout """
        ret = self.input_layer
        if use_dropout:
            num_str = NNl.get_num_streams(np.prod(self.out_shape))
            mask = NNl.gen_mask(self.srng, self.out_shape, self.p_retain,
                    num_str)
            ret *= mask / self.p_retain
        return ret


class OutputLayer(Layer):

    """ This layer will come last in most structure definitions of a network.
    The cost calculations for supervised training are done here, as well as
    the actual classification of data samples.

    Attributes:
        W: The matrix of weights
        b: The vector of biases
        params: A container for easy grouping of the layer's parameters

    The methods depend on only two main arguments:
        y: The labels for your dataset
        use_dropout: Toggles dropout usage for training vs testing, which is
                    propagated down through each input layer
    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        super(OutputLayer, self).__init__(input_layer, traits, "Output")
        self.out_shape = (Lshape[0], Lshape[1])
        self.W_shape = Lshape[1:]
        self.activation = activation
        self.l2decay = traits['l2decay']

        if len(Lshape) != 3:
            print("Logistic regression shape must be (2,), it is,", Lshape)

        # Initialize weights and biases (can load values later)
        self.W = NNl.gen_weights(rngs[0], self.W_shape, 0, traits['initW'])
        self.b = Tsh(np.zeros((Lshape[2],), dtype=Tfloat))
        self.params = [self.W, self.b]

    def p_y_given_x(self, use_dropout=True):
        """ Probability of a given state, using softmax, for classification """
        x = self.input_layer.output(use_dropout, depth=self.number)
        if x.ndim != 2:
            x = x.flatten(2)
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

    def mse(self, y, use_dropout=True):
        """ Calculates the mean squared error between the prediction
        and the label values.
        """
        x = self.input_layer.output(use_dropout, depth=self.number)
        if x.ndim != 2:
            x = x.flatten(2)
        # Activations, for use with regression
        y_act = self.activation(T.dot(x, self.W) + self.b)
        return T.mean(T.sqr(y_act - y))

    def log_loss(self, y, use_dropout=True):
        """ Calculates the negative log loss between the predicted and
        label values.
        """
        x = self.input_layer.output(use_dropout, depth=self.number)
        if x.ndim != 2:
            x = x.flatten(2)
        y_act = self.activation(T.dot(x, self.W) + self.b)
#       y_act = T.maximum(1e-15, T.minimum(1. - 1e-15, y_act))
        loss = -(y * T.log(y_act) + (1 - y) * T.log(1 - y_act))
        return T.mean(loss)

    def negative_log_likelihood(self, y, use_dropout=True):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        logprob = T.log(self.p_y_given_x(use_dropout))
        return -T.mean(logprob[T.arange(y.shape[0]), y])

    def class_pred(self, use_dropout=False):
        """ Predict classes by the one with max probability """
        return T.argmax(self.p_y_given_x(use_dropout), axis=1)

    def errors(self, y, use_dropout=False):
        """ Calculate the total number of classification errors """
        return T.mean(T.neq(self.class_pred(use_dropout), y))


class FCLayer(Layer):

    """ A generic fully-connected layer of a neural network.

    The FCLayer also has options that allow it to be trained in an
    autoencoder, and utilize dropout and weight decay.

    Methods:
        signal: The pre-activation signal, i.e. x.W + b
        output: What the next layer will see. Applies activation and dropout.
        prop_r: Propagate 'right': a simple version of 'signal'
        prop_l: Propogate 'left': reconstruct the input from output,
                i.e. y.W.T + b'. Used for autoencoder training.
        reconstruct_mse: Mean-squared error of reconstruction. Used during
                the training as an autoencoder.

    Attributes:
        tag: Signifier for the layer type.
        rng: numpy rng (used for initialization)
        srng: Theano rng stream (used for generating dropout masks).
        input_layer: The layer which feeds into this one.
        in_shape: The shape of the input to this layer.
        out_shape: The shape of output this layer produces.
        number: 'n' where this layer is the nth layer in your network,
                starting from the Input as 0.
        p_retain: Probability of retaining a neuron after dropout.
        l2decay: L2 decay constant for this layer.
        activation: The non-linearity to apply to x.W + b (def. ReLU)
        d_rec: Input without dropout applied, used by autoencoder.
        best_error: Tracks the recon error during autoencoder training.
                (The Model class tracks the supervised training error)
        W: Weight matrix
        b: bias matrix
        bp: 'inverse' bias matrix (applied during prop_l)
        params: Convenient wrapper of params for calculating the gradient.
        pt_params: As above, but for autoencoder training
    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        super(FCLayer, self).__init__(input_layer, traits, "FC")

        self.p_retain = (1. - traits['dropout'])
        self.rng = rngs[0]
        self.srng = rngs[1]
        self.out_shape = (Lshape[0], Lshape[2])
        self.W_shape = Lshape[1:]
        self.activation = activation
        self.l2decay = traits['l2decay']
        self.d_rec = input_layer.output(False)
        self.best_error = np.inf

        if len(Lshape) != 3:
            print "FC layer shape must be (2,), it is,", Lshape

        self.W = NNl.gen_weights(self.rng, self.W_shape, 0, traits['initW'])
        self.b = Tsh(np.zeros(Lshape[2], dtype=Tfloat))
        self.ib = Tsh(np.zeros(Lshape[1], dtype=Tfloat))
        self.params = [self.W, self.b,]
        self.pt_params = [self.W, self.b, self.ib]

    def signal(self, use_dropout=False, depth=0):
        """ Raw signal from applying weights and bias, pre-activation """
        if depth > 0:
            x = self.input_layer.output(use_dropout, depth=(depth-1))
        else:
            x = self.input_layer.output(False, depth=0)
        if x.ndim > 2:
            x = x.flatten(2)
        return T.dot(x, self.W) + self.b

    def output(self, use_dropout=False, depth=0):
        """ Apply the activation and dropout to the signal, producing
        output that will be used by subsequent layers
        """
        out = self.activation(self.signal(use_dropout=use_dropout, depth=depth))
        c_shape = self.out_shape
        if use_dropout:
            num_str = NNl.get_num_streams(np.prod(c_shape))
            mask = NNl.gen_mask(self.srng, c_shape, self.p_retain, num_str)
            out = out * mask / self.p_retain
        return out

    def prop_R(self, in_vectors):
        """ Simple version of 'signal' method: Right propagation """
        return T.dot(in_vectors, self.W) + self.b

    def prop_L(self, in_vectors):
        """ Apply the weight transpose and the inverse bias: Left prop. """
        return T.dot(in_vectors, self.W.T) + self.ib

    def reconstruct_mse(self, r_activation):
        """ Used for training as an auto-encoder, this compares the input
        and its reconstruction via forward and backward propagation.
        Args:
            r_activation: Activation function to apply during reconstruction.
                The choice for this depends on the distribution of your
                input, so the deeper hidden layers just use a soft ReLU in
                general, but the initial hidden layer will need to use a
                function dependent on your feature set.
        """
        x0 = self.input_layer.output()
        if x0.ndim > 2:
            x0 = x0.flatten(2)
        out = self.activation(self.signal(use_dropout=True, depth=1))
        xr = r_activation(self.prop_L(out))
        return T.mean(T.sum(T.sqr(x0 - xr), axis=1))


class ConvLayer(Layer):

    """ This layer applies the convolution step to input data. This means
    rastering a square matrix ("filter") across the input. Usually, many
    parallel filters applied per layer (nKernels). It is possible to pad
    the borders with zeros so the convop output is the same size as the
    input. This implementation uses the pylearn2 FilterActs method, based
    on Alex Krizhevsky's speedy CUDA ConvNet code.

    Methods:
        output: What the next layer will see. Applies activation and dropout.

    Attributes:
        tag: Signifier for the layer type.
        rng: numpy rng (used for initialization)
        input_layer: The layer which feeds into this one.
        number: 'n' where this layer is the nth layer in your network,
                starting from the Input as 0.
        l2decay: L2 decay constant for this layer.
        filter_shape: shape of the convolutional filter bank:
            (kernels, channels, filter size, filter size)
        pad: zeropadding for applying conv filter at all points
        W: Weight matrix
        params: Convenient wrapper of params for calculating the gradient.

    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        super(ConvLayer, self).__init__(input_layer, traits, "Conv")

        self.rng = rngs[0]
        self.l2decay = traits['l2decay']
        filter_shape = Lshape[1]
        # The number of input channels must match number of filter channels
        assert Lshape[0][1] == filter_shape[1]
        self.pad = traits['padding']

        self.W = NNl.gen_weights(self.rng, filter_shape, 0, traits['initW'])

        # convolve input feature maps with filters
        # Using Alex K.'s fast CUDA conv, courtesy of S. Dieleman
        self.x = self.input_layer.output(False)
        conv_op = FilterActs(pad=self.pad, partial_sum=1)
        input_shuffled = (self.x).dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_shuffled = (self.W).dimshuffle(1, 2, 3, 0) # bc01 to c01b
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_filters = gpu_contiguous(filters_shuffled)
        out_shuffled = conv_op(contiguous_input, contiguous_filters)
        self.conv_out = out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01

        # store parameters of this layer
        self.params = [self.W]

    def output(self, use_dropout=True, depth=0):
        """Just pass through for now"""
        return self.conv_out


class PoolLayer(Layer):

    """ This layer simply performs a MaxOut pooling, where a downsample
    factor N is specified, and for each NxN contiguous block of input the
    maximum value is taken as the output.
    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        super(PoolLayer, self).__init__(input_layer, traits, "Pool")

        self.pool_size = (traits['pool'], traits['pool'])
        self.activation = activation
        self.l2decay = traits['l2decay']

        self.b = Tsh(np.zeros((Lshape[1],), dtype=Tfloat))

        self.params = [self.b]

    def output(self, use_dropout=True, depth=0):
        """ Downsamples the input data and apply activation """
        conv_output = self.input_layer.output()
        input_shuffled = (conv_output).dimshuffle(1, 2, 3, 0)
        contiguous_input = gpu_contiguous(input_shuffled)
        pool_op = MaxPool(ds=3, stride=self.pool_size[0])
        out_shuffled = pool_op(contiguous_input)
        pool_out = out_shuffled.dimshuffle(3, 0, 1, 2)  # c01b to bc01

#       Theano routine
#       pool_out = downsample.max_pool_2d(input=self.input_layer.output(),
#                                       ds=self.pool_size, ignore_border=True)

        return self.activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
