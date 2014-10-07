""" The fully-connected layer class, which will be found in most network
architectures.
"""


import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

import NNlib as NNl

Tfloat = theano.config.floatX
Tsh = theano.shared
Tsig = T.nnet.sigmoid


class FCLayer(object):

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
        self.tag = "FC"
        self.number = traits['number']

        self.input_layer = input_layer
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
