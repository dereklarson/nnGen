import cPickle, gzip
import numpy as np

import theano
import theano.tensor as T

Tfloat = theano.config.floatX
Tsh = theano.shared

MAX_CHANNELS = 20

def GenWeights(rng, Wshape, mean, std):
    weights = rng.normal(mean, std, size=Wshape)
    return np.asarray(weights, dtype=Tfloat)

def GenMask(srng, out_shape, p_retain, num_str=1024):
    mask = srng.binomial(out_shape, p=p_retain, dtype='int32', nstreams=num_str)
    return T.cast(mask, Tfloat)

def GenAug(rng, jitter, flip):
    out1 = rng.randint(0, jitter+1, 2)
    out2 = rng.randint(0, 2, 1) * 2 - 1 if flip[0] else [1]
    out3 = rng.randint(0, 2, 1) * 2 - 1 if flip[1] else [1]
    return np.asarray(np.concatenate((out1, out2, out3)), dtype='int32')

def GetBatch_0(data, index, batch_size):
    batch = data[index * batch_size:(index + 1) * batch_size]
    return batch

def GetBatch(data, index, batch_size, in_dims, aug):
    batch = data[index * batch_size:(index + 1) * batch_size]
    x0, y0 = (aug[0], aug[1])
    x1, y1 = (in_dims[0] + x0, in_dims[1] + y0)
    batch = batch[:,:, y0:y1, x0:x1]  
    batch = batch[:,:, ::aug[2], ::aug[3]]  
    return batch

def JShift(curr_shape, shift):
    return curr_shape[:-2] + (curr_shape[-2] - shift, curr_shape[-1] - shift)

def tShuffle(data, srng):
    rows = data.shape[0]
    ret = T.permute_row_elements(data.T, rng.permutation(n=rows)).T
    return ret

def ProcessInfo(info, data_shape, cut, channels):
    if cut == -1:
        if 'cut' in info:
            cut = info['cut']
        else:
            cut = (9 * data_shape[0] / 10)
            cut = cut - cut % 128
    if channels == -1:
        if len(data_shape) == 2 or data_shape[1] > MAX_CHANNELS:
            print "Assuming no channel information"
            channels = 1
            shape_index = 1
        else:
            channels = data_shape[1]
            shape_index = 2
    in_dims = data_shape[shape_index:]
    return cut, channels, in_dims, info['description']

def shareloader(infile, do_valid=True, cut=-1, channels=-1):
    print "loading data...", 
    info, data, label = cPickle.load(open(infile, 'rb'))
    cut, channels, in_dims, desc = ProcessInfo(info, data.shape, cut, channels)
    out_dim = label.shape[1] if len(label.shape) > 1 else 1
    training = []
    if do_valid:
        validation = []
        training.append(Tsh(np.asarray(data[:cut], dtype=Tfloat)))
        training.append(Tsh(np.asarray(label[:cut], dtype=Tfloat)))
        validation.append(Tsh(np.asarray(data[cut:], dtype=Tfloat)))
        validation.append(Tsh(np.asarray(label[cut:], dtype=Tfloat)))
        print "done\n", desc
        print "###", len(data[:cut]), "training and", len(data[cut:]), "val. samples"
        return training, validation, desc, [channels, in_dims, out_dim]
    else:
        training.append(Tsh(np.asarray(data, dtype=Tfloat)))
        training.append(Tsh(np.asarray(label, dtype=Tfloat)))
        print "done\n", desc
        print "###", len(data), "test samples (no validation)"
        return training, None, desc, [channels, in_dims, out_dim]

def pm1(srng):
    return T.cast(srng.uniform() > 0.5, dtype=int32) * 2 - 1

def RandInt(srng, low=0, high=2, num_str=1024):
    out = srng.uniform(size=(1,), dtype=Tfloat, nstreams=num_str) * high + low
    return T.cast(T.floor(out), 'int32')

def GetNumStreams(size):
    ret = 2
    while (ret < size): ret *= 2
    return ret

def NiceTime(sec_in):
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

def ScaleTanh(x):
    return 1.7159 * T.tanh(0.666666 * x)

def RectScaleTanh(x):
    return T.maximum(0., T.minimum(1., 1.7159 * T.tanh(0.666666 * x)))

def ReLU(x):
    return T.maximum(0., x)
