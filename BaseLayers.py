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
        out = self.x
        if use_dropout:
            out = out.flatten(2)
            num_str = NNl.get_num_streams(self.out_shape[0])
            mask = NNl.gen_mask(self.srng, out.shape, self.p_retain, num_str)
            out = out * mask / self.p_retain
        return out

class OutputLayer(object):
    """ This layer will come last in most structure definitions of a network.
    The cost calculations for supervised training are done here, as well as 
    the actual classification of data samples.
    """
    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        self.tag = "Output"
        self.number = traits['number']
        self.input_layer = input_layer
        self.layer_shape = Lshape
        self.activation = activation

        if len(self.layer_shape) != 2:
            print "Logistic regression shape must be (2,), it is,", \
                    self.layer_shape.shape

        # Initialize weights and biases (can load values later)
        self.W = Tsh(np.zeros(self.layer_shape, dtype=Tfloat))
        self.b = Tsh(np.zeros((self.layer_shape[1],), dtype=Tfloat))
        self.Wd = Tsh(np.ones(self.layer_shape, dtype=Tfloat) * traits['decayW'])
        self.bd = Tsh(np.zeros((self.layer_shape[1],), dtype=Tfloat))
        self.params = [self.W, self.b]
        self.pdecay = [self.Wd, self.bd]

    def p_y_given_x(self, use_dropout=True):
        """ Probability of a given state, using softmax, for classification """
        x = self.input_layer.output(use_dropout) 
        if x.ndim != 2:
            x = x.flatten(2)
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

    def mse(self, y, use_dropout=True):
        """ Calculates the mean squared error between the prediction
        and the label values
        """
        x = self.input_layer.output(use_dropout)
        if x.ndim != 2:
           x = layer_x.flatten(2)
        # Activations, for use with regression
        self.y_act = self.activation(T.dot(x, self.W) + self.b) 
        return T.mean(T.sqr(self.y_act - y))

    def negative_log_likelihood(self, y, use_dropout=True):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x(use_dropout))[T.arange(y.shape[0]), y])

    def class_pred(self, use_dropout=False):
        """ Predict classes by the one with max probability """
        return T.argmax(self.p_y_given_x(use_dropout), axis=1)

    def errors(self, y, use_dropout=False):
        """ Calculate the total number of classification errors """
        return T.mean(T.neq(self.class_pred(use_dropout), y))
