""" This class houses the raw feature data and adds some structure and
extra functionality such as preparing validation splits and shuffling.
"""


import cPickle as cp
import numpy as np
import glob
import re

import theano

Tfloat = theano.config.floatX
Tsh = theano.shared


class Dataset(object):

    """Structures your data for use in the neural net.

    Methods:
        load_raw: Load in data from a local file (specially formatted).
        prep_validation: Divide your data into training/validation chunks.
        rpermute: Randomly shuffle your dataset.

    Attributes:
        raw: The raw feature data as loaded in after load_raw.
        labels: The ground truth labels loaded in by load_raw.
        description: A blurb loaded by load_raw, displayed on loading.
        batch_n: Size of data chunks for training (SGD).
        T: Container for the training data and labels after loading onto
            the GPU (shared variables).
        V: Container for the validation data and labels after loading onto
            the GPU (shared variables).
        n_samples: Total number of data samples loaded
        sample_dim: Dimensions of the data (should be 3-tuple)
        label_dim: Dimensions of the label set (1 for classification)
        ltype: Datatype of the label set (int32 for classification)
        v_params: A dict filled by prep_validation with some basic
                info needed for subsequen training.
        test: Flag set when not planning to train.
    """

    def __init__(self, infile=None, rng=None):
        self.raw = []
        self.labels = []
        self.description = ""
        self.batch_n = 128
        self.T = []
        self.V = []
        self.b_samples = []
        self.n_samples = 0
        self.sample_dim = (0,)
        self.label_dim = 0
        self.ltype = None
        self.v_params = {}
        self.tperm = []
        self.vperm = []
        self.test = False
        if infile:
            self.load_raw(infile)
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(0)

    def load_raw(self, train_prefix, validation=None):
        """ Pull in raw data from a file.

        This will read in a local file and prepare the contained data for
        use with nnGen. The file must be a pickled list with a dict of
        info preceding the raw data and labels.

        Args:
            infile: datafile described above
        """
        train_files = sorted(glob.glob(train_prefix + '*'))
        if len(train_files) == 1:
            print "Loading data from", train_prefix
        else:
            suffixes = [re.findall(r'\d+', i) for i in train_files]
            rangestr = '[{}-{}]'.format(min(suffixes)[0], max(suffixes)[0])
            print "Loading chunks from", train_prefix, rangestr
        for infile in train_files:
            in_dict = cp.load(open(infile, 'rb'))
            info = in_dict['info']
            if 'description' in info:
                self.description = info['description']
                print info['description']
            self.sample_dim = in_dict['data'].shape[1:]
            self.b_samples.append(in_dict['data'].shape[0])
            while len(self.sample_dim) < 3:
                self.sample_dim = (1,) + self.sample_dim
            in_dict['data'] = in_dict['data'].reshape((-1,) + self.sample_dim)
            self.raw.append(in_dict['data'])
            self.labels.append(in_dict['labels'])

        self.n_samples = np.sum(self.b_samples)

        if len(self.labels[0].shape) > 1:
            self.label_dim = self.labels[0].shape[1]
            self.ltype = Tfloat
        else:
            self.label_dim = 1
            self.ltype = np.int32

    def prep_validation(self, batch=-1, cut=-1, k=6):
        """Split dataset into training/validation.

        Args:
            batch: batch size, defaults to 128 as defined in __init__
            cut: The number of training samples desired. The remainder
                will be used for validation
            k: 'k' in k-fold validation. If no cut is specified, we default
                to splitting the data into 'k' chunks with one as validation

        Returns:
            A dict containing info that will be passed along for training.
        """

        if cut == -1:
            cut = ((k - 1) * self.n_samples / k)
#           cut = cut - cut % batch
        if batch == -1:
            batch = self.batch_n
        else:
            self.batch_n = batch
        v_samples = self.b_samples.pop()
        v_data = self.raw.pop()
        v_labels = self.labels.pop()
        while v_samples < (self.n_samples - cut):
            v_data = np.concatenate(v_data, self.raw.pop())
            v_labels = np.concatenate(v_labels, self.labels.pop())
            v_samples += self.b_samples.pop()
        diff = v_samples - (self.n_samples - cut)
        self.raw.append(v_data[:diff])
        self.labels.append(v_labels[:diff])
        self.b_samples.append(diff)
        v_samples -= diff
        v_data = v_data[diff:]
        v_labels = v_labels[diff:]
#       T_d, T_l, self.T_Id = self.rpermute(self.raw[0], self.labels[0])
#       V_d, V_l, self.V_Id = self.rpermute(v_data, v_labels)

        self.T = [Tsh(np.asarray(self.raw[0], dtype=Tfloat)),
                    Tsh(np.asarray(self.labels[0], dtype=self.ltype))]
        self.V = [Tsh(np.asarray(v_data, dtype=Tfloat)),
                    Tsh(np.asarray(v_labels, dtype=self.ltype))]
        Tlen, Vlen = self.n_samples - len(v_data), len(v_data)
        print "##", Tlen, "training and", Vlen, "validation samples"
        t_batches = [i // batch for i in self.b_samples]
        self.V_params = {'t_batches': t_batches, 'v_batches': Vlen / batch}
        return self.V_params

    def rpermute(self, data, labels):
        """ Shuffles the data and labels randomly. """
        rperm = self.rng.permutation(len(data))
        return data[rperm], labels[rperm], rperm

    def zeropad(self, pad):
        """ Pads zeros around the data with width pad """
        new_shape = (self.raw.shape[-2] + pad, self.raw.shape[-1] + pad)
        T_new = np.zeros(self.T[0].get_value().shape[:-2] + new_shape)
        V_new = np.zeros(self.V[0].get_value().shape[:-2] + new_shape)
        x0 = y0 = pad/2
        x1 = y1 = new_shape[-2] - pad/2
        T_new[:, :, y0:y1, x0:x1] = self.T[0].get_value()
        self.T[0] = Tsh(np.asarray(T_new, dtype=Tfloat))
        V_new[:, :, y0:y1, x0:x1] = self.V[0].get_value()
        self.V[0] = Tsh(np.asarray(V_new, dtype=Tfloat))

    def set_rng(self, rng):
        """ Set the dataset RNG """
        self.rng = rng
