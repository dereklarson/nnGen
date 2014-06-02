import cPickle
import gzip, os, sys, time
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous

Tfloat = theano.config.floatX
Tsh = theano.shared

class ConvLayer(object):
    """Layer of a convolutional network """

    def __init__(self, rngs, input_layer, shape_in, traits, activation):
        image_shape = shape_in[0]
        filter_shape = shape_in[1]
        assert image_shape[1] == filter_shape[1]
        self.input_layer = input_layer 
        self.rng = rngs[0]
        self.pad = traits['padding']

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        # temporary test
        if traits['initW'] < 0:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            W_bound = traits['initW']

        self.W = Tsh(np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=Tfloat))
        self.Wd = Tsh(np.ones(filter_shape, dtype=Tfloat) * traits['decayW'])

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
        self.pdecay = [self.Wd]

    def output(self, use_dropout=True):
        return self.conv_out

class PoolLayer(object):
    """Pooling layer of a convolutional network """
    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        self.input_layer = input_layer 
        self.layer_shape = Lshape
        self.pool_size = (traits['pool'], traits['pool'])
        self.activation = activation

        self.b = Tsh(np.zeros((Lshape[1],), dtype=Tfloat))
        self.bd = Tsh(np.zeros((Lshape[1],), dtype=Tfloat))

        self.params = [self.b]
        self.pdecay = [self.bd]

    def output(self, use_dropout=True):
        # downsample each feature map individually, using maxpooling
        pool_out = downsample.max_pool_2d(input=self.input_layer.output(),
                                        ds=self.pool_size, ignore_border=True)
        return self.activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
