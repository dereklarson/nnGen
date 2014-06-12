import numpy as np
import theano
import theano.tensor as T
import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared

class InputLayer(object):
    def __init__(self, rngs, layer_x, Lshape, traits, activation=None):
        self.srng = rngs[1]
        self.x = layer_x
        self.out_shape = Lshape
        self.p_retain = (1. - traits['dropout'])
        self.traits = traits

    def output(self, use_dropout=True):
        out = self.x
        if use_dropout:
            out = out.flatten(2)
            num_str = NNl.GetNumStreams(self.out_shape[0])
            mask = NNl.GenMask(self.srng, out.shape, self.p_retain, num_str)
            out = out * mask / self.p_retain
        return out

class OutputLayer(object):
    def __init__(self, rngs, input_layer, Lshape, traits, activation):
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
        # Probability of a given state, using softmax, for classification
        x = self.input_layer.output(use_dropout)
        if x.ndim != 2:
            x = x.flatten(2)
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

    def mean_square_error(self, y, use_dropout=True):
        """Return the mean squared error between the prediction
        and the label values"""
        x = self.input_layer.output(use_dropout)
        if x.ndim != 2:
           x = layer_x.flatten(2)
        # Activations, for use with regression
        self.y_act = self.activation(T.dot(x, self.W) + self.b) 
        return T.mean(T.sqr(self.y_act - y))

    def negative_log_likelihood(self, y, use_dropout=True):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution. """
        return -T.mean(T.log(self.p_y_given_x(use_dropout))[T.arange(y.shape[0]), y])

    def errors(self, y, use_dropout=False):
        # Find the class with maximal probability for prediction
        y_pred = T.argmax(self.p_y_given_x(use_dropout), axis=1)
        return T.mean(T.neq(y_pred, y))
