"""This module contains the Model class which defines neural network
architecture, via a loaded parameter file or saved checkpoint file.
"""


from __future__ import print_function
import sys
import numpy as np
import cPickle as cp
import theano.tensor as T

from Layers import *
from NNlib import ReLU, RSTanh, Sigmoid, j_shift, find_type, pad_shape

# These are keywords that must be used in the Structure file
dict_data = ["type", "traits", "activation", "nKernels", "filter_size", "pool",
            "neurons", "padding", "dropout", "initW", "l2decay", "flipX",
            "flipY", "jitter", "jitterX", "jitterY", "zeropad"]
LC = {'Output': OutputLayer, 'FC': FCLayer, 'Conv': ConvLayer,
        'Input': InputLayer, 'Pool': PoolLayer}
AC = {'ReLU': ReLU, 'RSTanh': RSTanh, 'Sigmoid': Sigmoid, 'Linear': None}


class Model(object):

    """Class for defining neural network architecture.

    Defines everything needed to train a neural network, with regard
    to the model architecture. Everything is specified by a parameter
    file ('structure' file), which is parsed by the methods herein.

    Attributes:
        rngs: stores the RNGs passed onto the layers and also
            used for data augmentation.
        jitterX / Y: # of points to be cropped from the data on each
            axis; e.g. jitter of 4 each will give 25 different ways
            to crop the incoming data.
        flipX / Y: Whether or not to allow flips along that axis.
        zeropad: Apply 'zeropad' extra width of zeros around the data.
            (with zeropad == jitter, the augmented data size will be
            equal to the raw data size)
        best_error: Tracks the best validation error during SGD backprop.
        x: Theano variable representing the data input. Most important is
            that it has the right dimensions (4 needed for convolution).
        x_shape: shape of x, which will generally be something like:
            (batch_size, channels, data_x, data_y). The last two will
            be modified by jitter and zeropadding as well.
        layers: Stores the layer instances themselves.
        layer_info: List of dicts storing info about each layer.
        params: Stores each layers parameters together, for easy load/save.
        out_layer: References the final layer of the network--generally
            an 'Output' layer involving classification.
        ref_aug: Reference augmentation, used for checking validation error
            while jitter and flipping is present--crops from the center.
        y: Theano variable used in error calculation, generally given to
            be the training or validation labels.
        cost: Function that defines the cost of the network.
        val_error: Function that defines the validation error of the net.

    Methods:
        read_structure: Parses the structure file into a list of dicts
            that define the network layerwise. Ends up in self.layer_info.
        generate_shapes: Runs through the layer_info to figure out what
            the shape of each layer should be, starting from the input.
        craft_layers: Creates the layer instances.
        layer_splash: Prints a blurb describing a layer.
        save_model: Stores the model params and layer info to file.
        load_model: Retrieves model params and layer info from file.
        param_status: Analysis routine, writes out the weight and bias
            statistics to file for each layer.
        gen_aug: Generates data augmentation parameters based on the model
            jitter and flip settings (actual aug done in NNl.get_batch).
    """

    def __init__(self, rngs, in_shape, out_dim, CPFile="model_cp",
            struc_file="Structure"):

        # Initialize variables
        self.rngs = rngs
        self.jitterX = 0
        self.jitterY = 0
        self.flipX = 0
        self.flipY = 0
        self.zeropad = 0
        self.best_error = np.inf

        if len(in_shape) == 2:
            self.x = T.fmatrix('x')
        elif len(in_shape) == 3:
            self.x = T.tensor3('x')
        elif len(in_shape) == 4:
            self.x = T.tensor4('x')

        self.x_shape = in_shape

        # Create structure from previous model or struc file
        self.layers = []
        self.params = []
        self.out_layer = self.x
        if struc_file == "":
            print("Using " + CPFile + " file to load structure and params")
            self.load_model(CPFile)
        else:
            print("Building model from " + struc_file)
            self.layer_info = self.read_structure(struc_file)
            self.generate_shapes()
            self.craft_layers()

        # Take into account structure information
        self.x_shape = self.x_shape[:2] + (self.x_shape[2] -
                self.jitterX + self.zeropad,
                self.x_shape[3] - self.jitterY + self.zeropad)
        self.ref_aug = [self.jitterX / 2, self.jitterY / 2, 1, 1]

        if out_dim == 1:
            self.y = T.ivector('y')
            self.cost = self.out_layer.negative_log_likelihood(self.y, True)
            self.val_error = self.out_layer.errors(self.y, False)
        else:
            self.y = T.fmatrix('y')
#           self.cost = self.out_layer.mse(self.y, True)
#           self.val_error = T.sqrt(self.out_layer.mse(self.y, False))
#           self.cost = self.out_layer.log_loss(self.y, True)
            self.cost = self.out_layer.mse(self.y, True)
            self.val_error = self.out_layer.log_loss(self.y, False)


    def read_structure(self, filename):
        """ Parses a structure file provided by 'filename'.

        Layers are denoted by starting a line with "##" and keywords
        are specified by the 'dict_data' module constant.

        returns a layer-wise list of dictionaries defining layers
        """

        full_dict = []
        ldict = {}
        with open(filename, 'r') as my_file:
            for line in my_file:
                line_in = line.strip().replace('\t', '').split(':')
                if len(line_in) == 0:
                    continue
                if len(line_in) == 2 and line_in[0] in dict_data:
                    ldict[line_in[0]] = find_type(line_in[1])
                elif line_in[0][0:2] == '##':
                    if len(ldict) > 0: full_dict.append(ldict)
                    ldict = {'dropout': 0., 'activation': 'Linear', 'initW': -1,
                            'l2decay': 0.00, 'jitter': 0}
#                           'l2decay': 0.00, 'jitter': 0, 'zeropad': 0}
            if len(ldict) > 0: full_dict.append(ldict)
        return full_dict

    def generate_shapes(self):
        """Determines the shapes required for each layer (W matrix, etc).

        In addition, it fills the secondary parameters out, such as those
        found in the 'traits' dictionary provided to each layer. 'tlist'
        defines which keywords go into the 'traits'
        """
        c_shape = self.x_shape
        for layer in self.layer_info:
            if layer['type'] == 'Input':
                if 'zeropad' in layer:
                    self.zeropad = layer['zeropad']
                    c_shape = pad_shape(c_shape, layer['zeropad'])
                if 'jitter' in layer:
                    c_shape = j_shift(c_shape, layer['jitter'], layer['jitter'])
                    self.jitterX = layer['jitter']
                    self.jitterY = layer['jitter']
                if 'jitterX' in layer:
                    c_shape = j_shift(c_shape, 0, layer['jitterX'])
                    self.jitterX = layer['jitterX']
                if 'jitterY' in layer:
                    c_shape = j_shift(c_shape, layer['jitterY'], 0)
                    self.jitterY = layer['jitterY']
                if 'flipX' in layer: self.flipX = layer['flipX']
                if 'flipY' in layer: self.flipY = layer['flipY']
                layer['shape'] = c_shape
                tlist = ('dropout', 'jitter', 'flipX', 'flipY')
            elif layer['type'] == 'FC' or layer['type'] == 'Output':
                layer['shape'] = (c_shape[0], np.prod(c_shape[1:]),
                        layer['neurons'])
                c_shape = (c_shape[0], layer['neurons'],)
                tlist = ('dropout', 'initW', 'l2decay')
            elif layer['type'] == 'Conv':
                if 'padding' not in layer: layer['padding'] = 0
                fs = layer['filter_size']
                pad = layer['padding']
                layer['shape'] = [c_shape, (layer['nKernels'],
                        c_shape[1], fs, fs)]
                out = (c_shape[2] - fs + 2 * pad + 1)
                c_shape = (c_shape[0], layer['nKernels'], out, out)
                tlist = ('filter_size', 'nKernels', 'padding',
                        'initW', 'l2decay')
            elif layer['type'] == 'Pool':
                ps = layer['pool']
                layer['shape'] = c_shape
                c_shape = c_shape[0:2] + (c_shape[2] / ps, c_shape[3] / ps)
                tlist = ('pool', 'l2decay')
            layer['traits'] = {i: layer[i] for i in tlist}

    def craft_layers(self):
        """ Creates instances of the layers.  """
        new_layer = self.x
        layer_number = 0

        self.layers = []
        for layer in self.layer_info:
            layer['traits']['number'] = layer_number
            new_layer = LC[layer['type']](self.rngs, new_layer, layer['shape'],
                    layer['traits'], AC[layer['activation']])
            self.layer_splash(layer)
            self.layers.append(new_layer)
            layer_number += 1

        self.out_layer = new_layer

        self.params = []
        self.params += new_layer.params
        for i in range(len(self.layers)-2, 0, -1):
            self.params += self.layers[i].params

    def layer_splash(self, layer, output=sys.stdout):
        """ Prints a blurb containing information about a layer. """
        if layer['type'] == "Input":
            print("> Input has shape {} and dropout: {}".format(
                    layer['shape'], layer['dropout']), file=output)
            print ("#  Augmentation -- jitter: {}  flipX: {}  flipY: {}".format(
                    (self.jitterX, self.jitterY), self.flipX, self.flipY),
                    file=output)
        elif layer['type'] == "Conv":
            print(("+ Convolution: {} filters {} padding: {} initW: {} "
                    "l2decay: {}").format(layer['shape'][1][1],
                    layer['shape'][1][2:], layer['padding'], layer['initW'],
                    layer['l2decay']), file=output)
        elif layer['type'] == "Pool":
            in_1 = layer['shape'][2]
            pool = layer['pool']
            print("V Pool: scale {} for output: {} channels x {}".format(
                    pool, layer['shape'][1], (in_1 / pool, in_1 / pool)),
                    file=output)
        elif layer['type'] == "FC":
            print(("* FCLayer with shape {}  initW: {}  l2decay: {} "
                    "dropout: {}").format(layer['shape'][1:], layer['initW'],
                    layer['l2decay'], layer['dropout']), file=output)
        elif layer['type'] == "Output":
            print("= Output neurons: {} initW: {}  l2decay: {}".format(
                    layer['shape'][1:], layer['initW'], layer['l2decay']),
                    file=output)
        else:
            print("? {} with shape {} initW: {}  l2decay: {}".format(
                    layer['type'], layer['shape'], layer['initW'],
                    layer['l2decay']), file=output)

    def save_model(self):
        """ Stores the model params and layer info to file. """
        cp.dump([[i.get_value() for i in self.params], self.layer_info,
                    self.best_error], open("model_cp", 'wb'), 2)

    def load_model(self, CPFile):
        """ Retrieves model params and layer info from file. """
        temp_params, self.layer_info, val_error = cp.load(open(CPFile))
        self.jitterX = self.jitterY = self.flipX = self.flipY = 0
        if 'jitter' in self.layer_info[0]:
            self.jitterX = self.layer_info[0]['jitter']
            self.jitterY = self.layer_info[0]['jitter']
        if 'jitterX' in self.layer_info[0]:
            self.jitterX = self.layer_info[0]['jitterX']
        if 'jitterY' in self.layer_info[0]:
            self.jitterY = self.layer_info[0]['jitterY']
        if 'flipX' in self.layer_info[0]:
            self.flipX = self.layer_info[0]['flipX']
        if 'flipY' in self.layer_info[0]:
            self.flipY = self.layer_info[0]['flipY']
        self.craft_layers()
        print('Previous best validation:', val_error)
        self.best_error = val_error
        for i in range(len(temp_params)):
            self.params[i].set_value(temp_params[i])

    def param_status(self, epoch, output=sys.stdout):
        """ Prints max, mean, and std of weights and biases, for analysis. """
        print("epoch: ", epoch, "___________", file=output)
        for layer in self.layers[1:]:
            info = "{}-{} | ".format(layer.number, layer.tag)
            if hasattr(layer, 'W'):
                w_max = np.absolute(layer.W.get_value()).max()
                w_avg = layer.W.get_value().mean()
                w_std = layer.W.get_value().std()
                info += "W: {:.6f} {:.6f} {:.6f} ".format(w_max, w_avg, w_std)
            if hasattr(layer, 'b'):
                b_max = np.absolute(layer.b.get_value()).max()
                b_avg = layer.b.get_value().mean()
                b_std = layer.b.get_value().std()
                info += "b: {:.6f} {:.6f} {:.6f}".format(b_max, b_avg, b_std)
            print(info, file=output)

    def gen_aug(self):
        """ Generates augmentation parameters: crop x / y, flip x / y """
        out1 = self.rngs[0].randint(0, self.jitterX + 1)
        out2 = self.rngs[0].randint(0, self.jitterY + 1)
        out3 = self.rngs[0].randint(0, 2) * 2 - 1 if self.flipX else 1
        out4 = self.rngs[0].randint(0, 2) * 2 - 1 if self.flipY else 1
        return np.asarray([out1, out2, out3, out4], dtype='int32')

