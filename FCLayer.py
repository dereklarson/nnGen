import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams

import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared
Tsig = T.nnet.sigmoid

class FCLayer(object):
    """ A generic fully-connected layer of a neural network, with options 
    that allow it to be trained in an autoencoder and utilize dropout
    """
    def __init__(self, rngs, input_layer, Lshape, traits, activation):
        self.tag = "FC"
        self.number = traits['number']

        self.input_layer = input_layer
        self.in_shape = (Lshape[0], )
        self.out_shape = (Lshape[1], )
        self.p_retain = (1. - traits['dropout'])
        self.rng = rngs[0]
        self.srng = rngs[1]
        self.activation = activation
        self.d_rec = input_layer.output(False)
        self.best_error = np.inf

        if len(Lshape) != 2:
            print "FC layer shape must be (2,), it is,", \
                    Lshape.shape

        if traits['initW'] < 0: traits['initW'] = 1./np.sqrt(Lshape[0])
        self.W = Tsh(NNl.gen_weights(self.rng, Lshape, 0, traits['initW']))
        self.b = Tsh(np.zeros(self.out_shape, dtype=Tfloat))
        self.ib = Tsh(np.zeros(self.in_shape, dtype=Tfloat))
        self.params = [self.W, self.b,]
        self.pt_params = [self.W, self.b, self.ib]
        self.Lnorm = np.float32(1./np.sqrt(np.prod(Lshape)))

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
        if use_dropout:
            num_str = NNl.get_num_streams(self.out_shape[0])
            self.mask = NNl.gen_mask(self.srng, self.out_shape, self.p_retain, num_str)
            out = out * self.mask / self.p_retain
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
        """
        x0 = self.input_layer.output()
        if x0.ndim > 2:
            x0 = x0.flatten(2)
        out = self.activation(self.signal(use_dropout=True, depth=1))
        xr = r_activation(self.prop_L(out))
        return T.mean(T.sum(T.sqr(x0 - xr), axis=1))
