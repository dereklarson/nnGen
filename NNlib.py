import cPickle as cp
import numpy as np

import theano
import theano.tensor as T

Tfloat = theano.config.floatX
Tsh = theano.shared

MAX_CHANNELS = 20

def gen_weights(rng, Wshape, mean, std):
    """ Simple Gaussian generator used to init layer weights """
    weights = rng.normal(mean, std, size=Wshape)
    return np.asarray(weights, dtype=Tfloat)

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
    x1, y1 = (in_dims[0] + x0, in_dims[1] + y0)
    batch = batch[:,:, y0:y1, x0:x1]  
    batch = batch[:,:, ::aug[3], ::aug[2]]  
#   batch = batch[:,:, y0:y1:aug[3], x0:x1:aug[2]]  
    return batch

def j_shift(curr_shape, shiftX, shiftY):
    """ Helper to modify the in_shape tuple by jitter amounts """
    return curr_shape[:-2] + (curr_shape[-2] - shiftY, curr_shape[-1] - shiftX)

def prepare_log(model, description, fname="log_train"):
    """ Open logfile and generate a header for it """
    logfile = open(fname, 'w')
    logfile.write("#Data: " + description + "\n")
    for layer in model.layer_info:
        logfile.write("#")
        model.layer_splash(layer, logfile)
    logfile.write("#time    epoch  LR      test      train     \n")
    return logfile

def get_num_streams(size):
    """ Find how many streams to use for Theano RNG """
    ret = 2
    while (ret < size): ret *= 2
    return ret

def nice_time(sec_in):
    """ Pretty formatting for time stamps """
    seconds = int(sec_in) % 60
    minutes = int((sec_in / 60)) % 60
    hours = int(sec_in / 3600)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def find_type(string_in):
    out = string_in
    try:
        out = int(string_in)
    except ValueError:
        try:
            out = float(string_in)
        except ValueError:
            None
    return out

def STanh(x):
    """ Scaled tanh based on Y. LeCun's early suggestion """
    return 1.7159 * T.tanh(0.666666 * x)

def RSTanh(x):
    """ Rectified version of the above """
    return T.maximum(0., T.minimum(1., 1.7159 * T.tanh(0.666666 * x)))

def ReLU(x):
    return T.maximum(0., x)

def SoftReLU(x):
    """ Version of the above with no cusps, for autoencoder training """
    return T.log(1 + T.exp(x))

def get_profiler(profiling, lang='c'):
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
