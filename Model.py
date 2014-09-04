from __future__ import print_function
import sys
import numpy as np
import cPickle as cp
import theano
import theano.tensor as T

from BaseLayers import InputLayer, OutputLayer
from FCLayer import FCLayer
from ConvLayers import ConvLayer, PoolLayer
from NNlib import ReLU, RSTanh, j_shift, find_type
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams

# These are keywords that must be used in the Structure file
dict_data = ["type", "traits", "activation", "nKernels", "filter_size", "pool",
            "neurons", "padding", "dropout", "initW", "decayW", "flipX",
            "flipY", "jitter", "jitterX", "jitterY"]
LC = {'Output': OutputLayer, 'FC': FCLayer, 'Conv': ConvLayer,
        'Input': InputLayer, 'Pool': PoolLayer}
AC = {'ReLU': ReLU, 'RSTanh': RSTanh, 'Linear': None}

# Class for creating neural network architechture
class Model:
    def __init__(self, rngs, in_shape, out_dim, CPFile="model_cp", struc_file="Structure"):

        self.rngs = rngs
        self.jitterX = 0
        self.jitterY = 0
        self.flipX = 0
        self.flipY = 0
        self.best_error = np.inf

        if len(in_shape) == 2:
            self.x = T.fmatrix('x')
        elif len(in_shape) == 3:
            self.x = T.tensor3('x')
        elif len(in_shape) == 4:
            self.x = T.tensor4('x')

        self.x_shape = in_shape

        if struc_file == "":
            print("Using " + CPFile + " file to load structure and params")
            self.load_model(CPFile)
        else:
            print("Building model from " + struc_file)
            self.layer_info = self.read_structure(struc_file)
            self.generate_shapes()
            self.craft_layers()

            self.x_shape = self.x_shape[:2] + (self.x_shape[2] - self.jitterX,
                            self.x_shape[3] - self.jitterY)

        if out_dim == 1:
            self.y = T.ivector('y')
            self.cost = self.out_layer.negative_log_likelihood(self.y, True)
            self.val_error = self.out_layer.errors(self.y, False)
        else:
            self.y = T.fmatrix('y')
            self.cost = self.out_layer.mse(self.y, True)
            self.val_error = T.sqrt(self.out_layer.mse(self.y, False))


    def read_structure(self, filename):
        full_dict = []
        ldict = {}
        with open(filename, 'r') as my_file:
            for line in my_file:
                line_in = line.strip().replace('\t','').split(':')
                if len(line_in) == 0:
                    continue
                if len(line_in) == 2 and line_in[0] in dict_data:
                    ldict[line_in[0]] = find_type(line_in[1])
                elif line_in[0][0:2] == '##':
                    full_dict.append(ldict) if len(ldict) > 0 else None
                    ldict = {'dropout': 0.0, 'activation': 'Linear', 'initW': -1,
                            'decayW': 0.00, 'jitter': 0}
            full_dict.append(ldict) if len(ldict) > 0 else None
        return full_dict

    def generate_shapes(self):
        curr_shape = self.x_shape
        for layer in self.layer_info:
            if layer['type'] == 'Input':
                if 'jitter' in layer:
                    curr_shape = j_shift(curr_shape, layer['jitter'], layer['jitter'])
                    self.jitterX = layer['jitter']
                    self.jitterY = layer['jitter']
                if 'jitterX' in layer:
                    curr_shape = j_shift(curr_shape, 0, layer['jitterX'])
                    self.jitterX = layer['jitterX']
                if 'jitterY' in layer:
                    curr_shape = j_shift(curr_shape, layer['jitterY'], 0)
                    self.jitterY = layer['jitterY']
                if 'flipX' in layer: self.flipX = layer['flipX']
                if 'flipY' in layer: self.flipY = layer['flipY']
                layer['shape'] = curr_shape
                tlist = ('dropout', 'jitter', 'flipX', 'flipY')
            elif layer['type'] == 'FC':
                if len(curr_shape) > 1:
                    layer['shape'] = (np.prod(curr_shape[1:]), layer['neurons'])
                else:
                    layer['shape'] = (curr_shape[0], layer['neurons'])
                curr_shape = (layer['neurons'],)
                tlist = ('dropout', 'initW', 'decayW')
            elif layer['type'] == 'Conv':
                if 'padding' not in layer: layer['padding'] = 0
                fs = layer['filter_size']
                pad = layer['padding']
                layer['shape'] = [curr_shape, (layer['nKernels'], curr_shape[1], fs, fs)]
                out = (curr_shape[2] - fs + 2 * pad + 1)
                curr_shape = (curr_shape[0], layer['nKernels'], out, out)
                tlist = ('filter_size', 'nKernels', 'padding', 'initW', 'decayW')
            elif layer['type'] == 'Pool':
                ps = layer['pool']
                layer['shape'] = curr_shape
                curr_shape = curr_shape[0:2] + (curr_shape[2] / ps, curr_shape[3] / ps)
                tlist = ('pool', )
            elif layer['type'] == 'Output':
                if len(curr_shape) > 1:
                    layer['shape'] = (np.prod(curr_shape[1:]), layer['neurons'])
                else:
                    layer['shape'] = (curr_shape[0], layer['neurons'])
                curr_shape = (layer['neurons'],)
                tlist = ('dropout', 'initW', 'decayW')
            layer['traits'] = {i: layer[i] for i in tlist}

    def craft_layers(self):
        self.layers = []
        new_layer = self.x
        layer_number = 0

        for layer in self.layer_info:
            layer['traits']['number'] = layer_number
            new_layer = LC[layer['type']](self.rngs, new_layer, layer['shape'], \
                    layer['traits'], AC[layer['activation']])
            self.layer_splash(layer)
            self.layers.append(new_layer)
            layer_number += 1

        self.out_layer = new_layer

        self.params = new_layer.params
        for i in range(len(self.layers)-2, 0, -1):
            self.params += self.layers[i].params

    def layer_splash(self, layer, output=sys.stdout):
        if layer['type'] == "Input":
            print("> Input has shape {} and dropout: {}".format( \
                    layer['shape'], layer['dropout']), file=output)
            print ("#  Augmentation -- jitter: {}  flipX: {}  flipY: {}".format(\
                    (self.jitterX, self.jitterY), self.flipX, self.flipY), file=output)
        elif layer['type'] == "Conv":
            print("+ Convolution: {} filters {} padding: {} initW: {}  decayW: {}".format( \
                    layer['shape'][1][1], layer['shape'][1][2:], 
                    layer['padding'], layer['initW'], layer['decayW']), file=output)
        elif layer['type'] == "Pool":
            in_1 = layer['shape'][2]
            pool = layer['pool']
            print("V Pool: scale {} for output: {} channels x {}".format( \
                    pool, layer['shape'][1], (in_1 / pool, in_1 / pool)), file=output)
        elif layer['type'] == "FC":
            print("* FCLayer with shape {}  initW: {}  decayW: {}  dropout: {}".format( \
                    layer['shape'], layer['initW'], layer['decayW'], layer['dropout']), file=output)
        elif layer['type'] == "Output":
            print("= Output neurons: {} initW: {}  decayW: {}".format( \
                    layer['shape'], layer['initW'], layer['decayW']), file=output)
        else:
            print("? {} with shape {} initW: {}  decayW: {}".format( \
                    layer['type'], layer['shape'], layer['initW'],
                    layer['decayW']), file=output)

    def save_model(self, val_error):
        cp.dump([[i.get_value() for i in self.params], self.layer_info, val_error], \
                open("model_cp", 'wb'), 2)

    def load_model(self, CPFile):
        temp_params, self.layer_info, val_error = cp.load(open(CPFile))
        self.jitterX = self.jitterY = self.flipX = self.flipY = 0
        if 'jitter' in self.layer_info[0]:
            self.jitterX = self.layer_info[0]['jitter']
            self.jitterY = self.layer_info[0]['jitter']
        if 'jitterX' in layer:
            self.jitterX = layer['jitterX']
        if 'jitterY' in layer:
            self.jitterY = layer['jitterY']
        if 'flipX' in self.layer_info[0]:
            self.flipX = self.layer_info[0]['flipX']
        if 'flipY' in self.layer_info[0]:
            self.flipY = self.layer_info[0]['flipY']
        self.craft_layers()
        print('Previous best validation:', val_error)
        self.best_error = val_error
        for i in range(len(temp_params)):
            self.params[i].set_value(temp_params[i])

    def gen_aug(self):
        out1 = self.rngs[0].randint(0, self.jitterX + 1)
        out2 = self.rngs[0].randint(0, self.jitterY + 1)
        out3 = self.rngs[0].randint(0, 2) * 2 - 1 if self.flipX else 1
        out4 = self.rngs[0].randint(0, 2) * 2 - 1 if self.flipY else 1
        return np.asarray([out1, out2, out3, out4], dtype='int32')

