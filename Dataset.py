import cPickle as cp
import numpy as np

import theano
import theano.tensor as T

Tfloat = theano.config.floatX
Tsh = theano.shared

class Dataset:
    def __init__(self, infile=None):
        self.raw = None
        self.labels = None
        self.description = ""
        if infile:
            self.load_raw(infile)

    def load_raw(self, infile):
        print "Loading data from", infile
        info, self.raw, self.labels = cp.load(open(infile, 'rb'))
        if 'description' in info:
            self.description = info['description']
            print info['description']
        self.n_samples = self.raw.shape[0]
        self.sample_dim = self.raw.shape[1:]
        if len(self.labels.shape) > 1:
            self.label_dim = self.labels.shape[1]
            self.ltype = Tfloat
        else:
            self.label_dim = 1
            self.ltype = np.int32

    def prep_validation(self, batch=128, cut=-1, k=6):
        if cut == -1:
            cut = ((k - 1) * self.n_samples / k)
            cut = cut - cut % batch
        self.T = [Tsh(np.asarray(self.raw[:cut], dtype=Tfloat)),
                    Tsh(np.asarray(self.labels[:cut], dtype=self.ltype))]
        self.V = [Tsh(np.asarray(self.raw[cut:], dtype=Tfloat)),
                    Tsh(np.asarray(self.labels[cut:], dtype=self.ltype))]
        Tl = len(self.raw[:cut])
        Vl = len(self.raw[cut:])
        print "##", Tl, "training and", Vl, "validation samples"
        ret = {'t_batches': Tl / batch, 'v_batches': Vl / batch}
        return ret

