import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams
import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared

class FCLayer(object):
    def __init__(self, rngs, input_layer, Lshape, traits, activation):

        self.input_layer = input_layer
        self.out_shape = (Lshape[1], )
        self.p_retain = (1. - traits['dropout'])
        self.rng = rngs[0]
        self.srng = rngs[1]
        self.activation = activation

        if len(Lshape) != 2:
            print "FC layer shape must be (2,), it is,", \
                    Lshape.shape

        if traits['initW'] < 0: traits['initW'] = 0.01
        self.W = Tsh(NNl.GenWeights(self.rng, Lshape, 0, traits['initW']))
        self.b = Tsh(np.zeros(self.out_shape, dtype=Tfloat))
        self.Wd = Tsh(np.ones(Lshape, dtype=Tfloat) * traits['decayW'])
        self.bd = Tsh(np.zeros(self.out_shape, dtype=Tfloat))
        self.params = [self.W, self.b]
        self.pdecay = [self.Wd, self.bd]

    def output(self, use_dropout=True):
        x = self.input_layer.output(use_dropout)
        if x.ndim > 2:
            x = x.flatten(2)
        out = self.activation(T.dot(x, self.W) + self.b)
        if use_dropout:
            self.mask = NNl.GenMask(self.srng, self.out_shape, self.p_retain)
            out = out * self.mask / self.p_retain
        return out
