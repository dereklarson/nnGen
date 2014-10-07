""" This file contains two types of layers:  convolutional and pooling.
These will be useful for applications in which features are locally
correlated with their neighbors (notably images) and one usually applies
a convolutional layer first and then a subsequent pooling layer.
"""


import numpy as np

import theano
from theano.tensor.signal import downsample
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared


class ConvLayer(object):

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
        self.tag = "Conv"
        self.rng = rngs[0]
        self.input_layer = input_layer
        self.number = traits['number']
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

class PoolLayer(object):

    """ This layer simply performs a MaxOut pooling, where a downsample
    factor N is specified, and for each NxN contiguous block of input the
    maximum value is taken as the output.
    """

    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        self.tag = "Pool"
        self.number = traits['number']
        self.input_layer = input_layer
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
