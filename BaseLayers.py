""" This file contains the Input and Output layer classes, which will
necessary for your network architecture to function as is.
"""


import numpy as np

import theano
import theano.tensor as T

import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared

class InputLayer(object):

    """ This layer will come first in any structure definition of a network.
    It is involved in applying data augmentation, such as dropout, jitter
    and flipping.
    """

    def __init__(self, rngs, layer_x, Lshape, traits, activation=None):
        self.tag = "Input"
        self.number = traits['number']
        self.srng = rngs[1]
        self.x = layer_x
        self.out_shape = Lshape
        self.p_retain = (1. - traits['dropout'])
        self.traits = traits

    def output(self, use_dropout=False, depth=0):
        """ Provides data to next layer and applies dropout """
        ret = self.x
        if use_dropout:
            num_str = NNl.get_num_streams(np.prod(self.out_shape))
            mask = NNl.gen_mask(self.srng, self.out_shape, self.p_retain,
                    num_str)
            ret *= mask / self.p_retain
        return ret

class OutputLayer(object):

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
        self.tag = "Output"
        self.number = traits['number']
        self.input_layer = input_layer
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
        and the label values
        """
        x = self.input_layer.output(use_dropout, depth=self.number)
        if x.ndim != 2:
            x = x.flatten(2)
        # Activations, for use with regression
        y_act = self.activation(T.dot(x, self.W) + self.b)
        return T.mean(T.sqr(y_act - y))

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
