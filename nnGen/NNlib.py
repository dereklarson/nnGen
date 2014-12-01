"""This is a collection of miscellaneous helper functions, which cover such
tasks as grabbing data batches, providing activation functions, etc
"""


import numpy as np

import theano
import theano.tensor as T

Tfloat = theano.config.floatX
Tsh = theano.shared

MAX_CHANNELS = 20


def gen_weights(rng, w_shape, mean, std):
    """ Simple Gaussian generator used to init layer weights """
    if std == 0:
        return np.zeros(size=w_shape, dtype=Tfloat)
    elif std < 0:
        if len(w_shape) == 2:
            std = 1. / np.sqrt(w_shape[0])
        if len(w_shape) == 4:
            std = 1. / np.sqrt(np.prod(w_shape[1:]))
    weights = rng.normal(mean, std, size=w_shape)
    return Tsh(np.asarray(weights, dtype=Tfloat))

def gen_mask(srng, out_shape, p_retain, n_str=1024):
    """ Sets binary masks for use in dropout """
    mask = srng.binomial(out_shape, p=p_retain, dtype='int32', nstreams=n_str)
    return T.cast(mask, Tfloat)

def get_batch_0(data, index, batch_size):
    """ Return a slice of data for SGD batching (labels)"""
    batch = data[index * batch_size:(index + 1) * batch_size]
    return batch

def get_batch(data, index, batch_size, in_dims, aug):
    """ Return a slice of augmented data (jitter, flip) for SGD batching """
    batch = data[index * batch_size:(index + 1) * batch_size]
    x0, y0 = (aug[0], aug[1])
    x1, y1 = (in_dims[1] + x0, in_dims[0] + y0)
    batch = batch[:, :, y0:y1, x0:x1]
    batch = batch[:, :, ::aug[3], ::aug[2]]
    return batch

def j_shift(curr_shape, shiftX, shiftY):
    """ Helper to modify the in_shape tuple by jitter amounts """
    return curr_shape[:-2] + (curr_shape[-2] - shiftY, curr_shape[-1] - shiftX)

def pad_shape(curr_shape, pad):
    """ Pads zeros surrounding the data, pad must be even (pad/2 per side) """
    return curr_shape[:-2] + (curr_shape[-2] + pad, curr_shape[-1] + pad)

def prepare_log(model, description, fname="log_train"):
    """ Open logfile and generate a header for it """
    logfile = open(fname, 'w')
    logfile.write("#Data: " + description + "\n")
    for layer in model.layer_info:
        logfile.write("#")
        model.layer_splash(layer, logfile)
    logfile.write("#time    epoch  LR      train     test      \n")
    return logfile

def get_num_streams(size):
    """ Find how many streams to use for Theano RNG """
    #Optimized by hand, at least for small networks
    ret = size / 8
    return ret

def nice_time(sec_in):
    """ Pretty formatting for time stamps """
    seconds = int(sec_in) % 60
    minutes = int((sec_in / 60)) % 60
    hours = int(sec_in / 3600)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def find_type(string_in):
    """ Checks for int/float before settling on string """
    out = string_in
    try:
        out = int(string_in)
    except ValueError:
        try:
            out = float(string_in)
        except ValueError:
            pass
    return out

def Sigmoid(x):
    """ Sigmoid activation function """
    return T.nnet.sigmoid(x)

def STanh(x):
    """ Scaled tanh based on Y. LeCun's early suggestion """
    return 1.7159 * T.tanh(0.666666 * x)

def RSTanh2(x):
    """ Rectified version of the above """
    return T.maximum(0., T.minimum(1., 1.7159 * T.tanh(0.666666 * x)))

def RSTanh(x):
    """ Modified version of the above """
    return T.maximum(0., T.minimum(1., 0.858 * T.tanh(0.666666 * x) + 0.5))

def ReLU(x):
    """ Rectified linear unit """
    return T.maximum(0., x)

def SoftReLU(x):
    """ Version of the above with no cusps, for autoencoder training """
    return T.log(1 + T.exp(x))

def get_profiler(profiling, lang='c'):
    """ Figures out which profiling linker to use """
    if profiling:
        if lang == 'c':
            profmode = theano.ProfileMode(optimizer='fast_run',
                                        linker=theano.gof.OpWiseCLinker())
        elif lang == 'py':
            profmode = theano.ProfileMode(optimizer='fast_run',
                                        linker=theano.gof.PerformLinker())
    else:
        profmode = None
    return profmode
