#!/usr/bin/env python
"""This is the central file for running training instances of neural nets.
The core function "train_NN" will read in a dataset and structure file,
as well as any other command line arguments, and proceed to define and
train a neural net via autoencoding and/or SGD backpropagation.

It can be run as an executable and includes command line parsing.

For usage, run $ ./Main.py -h
"""


import getopt
import os
import sys
import numpy as np
import cPickle as cp

from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams

import NNlib as NNl                 #Helper library with useful functions
from Model import Model             #This is the class for NN architecture
from Dataset import Dataset
from Train import create_functions, create_functions_ae, train, class_probs


def train_NN(CPFile="", datafile="rawdata", dataset=None, SFile="structure",
            train_ae=False, profiling=False, rho=0.9, LR=0.001, n_epochs=500,
            batch_size=128, cut=-1, cv_k=10, seed=1000, predict=False,
            verbose=True):
    """The core routine for neural net training.

    Args:
        CPFile: Checkpoint file from which to resume a training run.
            Checkpoints are saved automatically as progress is made on a
            validation set, with standard filename "model_cp"
        datafile: File from which to retrieve raw data. This is subsequently
            loaded into a Dataset instance.
        dataset: Specifies a dataset to load directly. For use with training
            meta-algorithms that modify the dataset over multiple runs.
        SFile: The structure file that specifies the neural net architecture.
        train_ae: Flag for training as an autoencoder.
        profiling: Flag for turning on profiling for examining performance.
        rho: Momentum parameter. Standard momentum is used by default for
            both autoencoder and backprop training.
        LR: Learning rate.
        n_epochs: Number of epochs for training.
        batch_size: SGD mini-batch size. Processing speed increases for
            larger sizes, but fewer updates are made as a tradeoff.
        cut: Number of training examples to use from the raw data, with
            the rest as validation. '-1' indicates look at cv_k.
        cv_k: 'k' in K-fold validation. 1/k of the data used as a validation
            set, with the rest as training.
        seed: specifies random seed. For a given seed, dataset, and neural
            net architecture, the run will be repeatable.
        verbose: Flag determining whether to send continual updates to stdout.

    """

    sched_dict = {20: 0.010, 100: 0.001, 200: 0.0001}

    # This is passed to the theano functions for profiling
    profmode = NNl.get_profiler(profiling)

    # A dictionary collecting the necessary training parameters
    train_params = {'LR': LR, 'n_epochs': n_epochs, 'rho': rho,
                'verb': verbose, 'LRsched': sched_dict}

    # Create RNGs, one normal one Theano, which are passed to the Builder
    rng = np.random.RandomState(seed)
    theano_rng = MRG_RStreams(rng.randint(999999))
    rngs = [rng, theano_rng]

    # Load the dataset, then split for validation
    if dataset:
        data = dataset
        if not data.T:
            train_params.update(
                    data.prep_validation(batch=batch_size, cut=cut, k=cv_k))
        else:
            train_params.update(data.V_params)
    else:
        data = Dataset(datafile, rng)
        if predict:
            cv_k = 1
        train_params.update(
                data.prep_validation(batch=batch_size, cut=cut, k=cv_k))

    #*** CREATE A MODEL CLASS INSTANCE ***#

    in_shape = (batch_size,) + data.sample_dim

    # Load the checkpoint if there, otherwise use 'structure' to define network
    if os.path.isfile(CPFile):
        mymodel = Model(rngs, in_shape, data.label_dim, CPFile, struc_file="")
    else:
        mymodel = Model(rngs, in_shape, data.label_dim, struc_file=SFile)

    if mymodel.zeropad > 0:
        data.zeropad(mymodel.zeropad)

    #*** AUTOENCODER ***#
    #___________________#

    layers_to_train = []
    if train_ae:
        for layer in mymodel.layers:
            if layer.tag == "FC":
                layers_to_train.append(layer)

    for layer in layers_to_train:
        if layer.input_layer.tag == "Input":
            print "@ Autoencoding layer", layer.number, "with RSTanh"
            activ = NNl.RSTanh
        else:
            print "@ Autoencoding layer", layer.number, "with SoftReLU"
            activ = NNl.SoftReLU

        functions = create_functions_ae(layer, data.T, activ, batch_size,
                    rho, mymodel.x, mymodel.x_shape[2:], profmode)

        train_params['logfile'] = NNl.prepare_log(mymodel, data.description)
        train_params['error'] = layer

        train(mymodel, functions, train_params)

    #*** SGD BACKPROP ***#
    #____________________#

    if predict:
        print '@ Predicting'
#       predict_label(mymodel, data, train_params)
        cp.dump(class_probs(mymodel, data, train_params),
                open("class_p", 'wb'), 2)

    else:
        print '@ Training with SGD backprop'
        T_functions = create_functions(mymodel, data, rho, profmode)

        # Logfile made for analysis of training
        train_params['logfile'] = NNl.prepare_log(mymodel, data.description)
        train_params['error'] = mymodel

        train(mymodel, data, T_functions, train_params)

        print "\nBest validation: ", mymodel.best_error

    if profiling:
        profmode.print_summary()

#   mymodel.update_model("model_cp")
    return mymodel

# Here we can use the file as an executable and pass command line options
std_opts = {'-l': 'LR', '-s': 'seed', '-n': 'n_epochs', '-f': 'CPFile',
        '-d': 'datafile', '-S': 'SFile', '-b': 'batch_size', '-c': 'cut',
        '-C': 'cv_k'}

def usage():
    """Displays command line options
    """
    print "<exec> [ -h\tDisplay usage"
    print "\t-V\tnon-verbose"
    print "\t-P\tProfile mode"
    print "\t-p\tpredict using input data and checkpoint"
    print "\t-A\ttrain autoencoder"
    for key, value in std_opts.iteritems():
        print "\t", key, "\t<", value, ">"
    sys.exit()

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hVPpAl:s:n:f:d:S:b:c:C:")
    except getopt.GetoptError as err:
        print str(err)
        usage()
    pass_in = {}
    for opt, val in opts:
        if opt in std_opts:
            pass_in[std_opts[opt]] = NNl.find_type(val)
        elif opt == "-V":
            pass_in['verbose'] = False
        elif opt == "-P":
            pass_in['profiling'] = True
        elif opt == "-p":
            pass_in['predict'] = True
        elif opt == "-A":
            pass_in['train_ae'] = True
        elif opt in ("-h", "--help"):
            usage()
        else:
            assert False, "unhandled option"
    print "Passed arguments: ", pass_in
    train_NN(**pass_in)
