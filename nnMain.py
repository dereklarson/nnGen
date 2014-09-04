#!/usr/bin/env python
import getopt
import os
import sys
import time
import numpy as np
import cPickle as cp

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams

import NNlib as NNl                 #Helper library with useful functions
from Model import Model             #This is the class for NN architecture
from Dataset import Dataset
from Train import create_functions, create_functions_ae, train

def train_NN(CPFile="", datafile="dataset", SFile="structure", TrainAE=False,
            profiling=False, rho=0.9, LR=0.001, n_epochs=100, batch_size=128,
            seed=1000):
    ''' '''

    # This is passed to the theano functions for profiling
    profmode = NNl.get_profiler(profiling)

    # A dictionary collecting the necessary training parameters
    train_params = {'LR': LR, 'n_epochs': n_epochs, 'rho': rho}

    # Load the dataset, then split for validation
    data = Dataset(datafile)
    train_params.update(data.prep_validation(batch=batch_size, k=6))

    # Create RNGs, one normal one Theano, which are passed to the Builder
    rng = np.random.RandomState(seed)
    theano_rng = MRG_RStreams(rng.randint(999999))
    rngs = [rng, theano_rng]

    #*** CREATE A MODEL CLASS INSTANCE ***#

    in_shape = (batch_size,) + data.sample_dim

    # Load the checkpoint if there, otherwise use 'structure' to define network
    if os.path.isfile(CPFile):
        mymodel = Model(rngs, in_shape, data.label_dim, CPFile, struc_file="")
    else:
        mymodel = Model(rngs, in_shape, data.label_dim, struc_file=SFile)

    #*** AUTOENCODER ***#
    #___________________#

    for layer in mymodel.layers:
        if layer.tag == "FC" and TrainAE:
            params = layer.pt_params
            if layer.input_layer.tag == "Input":
                print "@ Autoencoding layer", layer.number, "with RSTanh"
                functions = create_functions_ae(layer, data.T, NNl.RSTanh,
                    batch_size, rho, mymodel.x, mymodel.x_shape[2:], profmode)
            else:
                print "@ Autoencoding layer", layer.number, "with SoftReLU"
                functions = create_functions_ae(layer, data.T, NNl.SoftReLU,
                    batch_size, rho, mymodel.x, mymodel.x_shape[2:], profmode)

            train_params['logfile'] = NNl.prepare_log(mymodel, data.description)
            train_params['error'] = layer

            train(mymodel, functions, train_params)

    #*** SGD BACKPROP ***#
    #____________________#

    print '@ Training with SGD backprop'
    T_functions = create_functions(mymodel, data.T, data.V, 
                                batch_size, rho, profmode)

    # Logfile made for analysis of training
    train_params['logfile'] = NNl.prepare_log(mymodel, data.description)
    train_params['error'] = mymodel

    train(mymodel, T_functions, train_params)

    if profiling: 
        profmode.print_summary()

# Here we can use the file as an executable and pass command line options
std_opts = {'-l': 'LR', '-s': 'seed', '-n': 'n_epochs', '-f': 'CPFile', 
        '-d': 'datafile', '-S': 'SFile', '-b': 'batch_size'}

def usage():
    print "<exec> [ -h\tDisplay usage"
    for key, value in std_opts.iteritems():
        print "\t", key, "\t<", value, ">"
    sys.exit()

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hl:s:n:f:d:S:b:")
    except getopt.GetoptError as err:
        print str(err) 
        usage()
    pass_in = {}
    for opt, val in opts:
        if opt in std_opts:
            pass_in[std_opts[opt]] = NNl.find_type(val)
        elif opt in ("-h", "--help"):
            usage()
        else:
            assert False, "unhandled option"
    print "Passed arguments: ", pass_in
    train_NN(**pass_in)
