"""This module contains the functions to create the Theano functions necessary
for training the neural net, either as an autoencoder or with SGD backprop. It
also contains the 'Train' function itself which performs the actual training.
"""


import time
import numpy as np

import theano
import theano.tensor as T

import NNlib as NNl                 #Helper library with useful functions


def create_functions(model, data, rho, profmode):
    """Creates Theano functions for SGD backprop.

    Args:
        data: Dataset instance on which to train.
        rho: momentum parameter
        profmode: used only for profiling
    """

    print 'Compiling functions...',

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    lrate = T.fscalar('lrate')  # learning rate
    aug = T.ivector('aug')      # data augmentation (jitter, flip)
    in_dims = model.x_shape[2:]
    x = model.x
    y = model.y

    # Functions to calculate our training and validation error while we run
    functions = {}
    f_input = [index, aug]

    functions['train_E'] = theano.function(f_input, model.val_error, givens={
            x: NNl.get_batch(data.T[0], index, data.batch_n, in_dims, aug),
            y: NNl.get_batch_0(data.T[1], index, data.batch_n)}, mode=profmode)

    if not data.test:
        functions['val_E'] = theano.function(f_input, model.val_error, givens={
            x: NNl.get_batch(data.V[0], index, data.batch_n, in_dims, aug),
            y: NNl.get_batch_0(data.V[1], index, data.batch_n)}, mode=profmode)

    train_input = [index, lrate, aug]

    # Functions to update the model, via momentum-based SGD
    mom_updates = []
    p_updates = []
    for layer in model.layers[-1:0:-1]:
        grads = T.grad(model.cost, layer.params)

        for grad_i, param_i in zip(grads, layer.params):
            delta_i = theano.shared(param_i.get_value() * 0.)
            c_update = (delta_i, rho * delta_i - lrate * grad_i -
                        lrate * layer.l2decay * param_i)
            mom_updates.append(c_update)
            p_updates.append((param_i, param_i + delta_i))
    functions['momentum'] = theano.function(train_input, model.cost,
            updates=mom_updates, givens={
            x: NNl.get_batch(data.T[0], index, data.batch_n, in_dims, aug),
            y: NNl.get_batch_0(data.T[1], index, data.batch_n)}, mode=profmode)
    functions['update'] = theano.function([], updates=p_updates, mode=profmode)

    print 'done'

    return functions

def create_functions_ae(layer, training, activation, batch_size,
        rho, x, in_dims, profmode):
    """Creates the Theano functions necessary to train as an autoencoder.

    Args:
        layer: The layer (FC) on which training will occur.
        training: The set of training data (no labels, just features).
        activation: The function to use for 'reconstruction' activation. This
            will generally align with the distribution found in the input you
            are trying to reconstruct. For example, for mean-std normalized
            input data, you might use a Tanh function in your 1st layer. If
            that first layer has a ReLU neuronal activation, then your 2nd
            hidden layer would probably use the SoftReLU reconstruction
            activation.
        batch_size: size of the batches used for autoencoder training.
        rho: momentum parameter
        x: Theano variable input to the current layer.
        in_dims: shape of the input piped through 'x'.
        profmode: Flag for profiling

    Returns:
        Dictionary of functions for autoencoder training: cost, and 2 updates
    """

    print 'Compiling autoencoder functions...',

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    lrate = T.fscalar('lrate')  # learning rate
    aug = T.ivector('aug')      # data augmentation (jitter, flip)

    # Functions to calculate our training and validation error while we run
    functions = {}

    cost = layer.reconstruct_mse(activation)

    f_input = [index, aug]

    functions['train_E'] = theano.function(f_input, cost, givens={
            x: NNl.get_batch(training[0], index, batch_size, in_dims, aug)},
            mode=profmode)

    # create a list of gradients for all model parameters
    grads = T.grad(cost, layer.pt_params)

    train_input = [index, lrate, aug]

    # Functions to update the model, via momentum-based SGD
    mom_updates = []
    p_updates = []
    for grad_i, param_i in zip(grads, layer.pt_params):
        delta_i = theano.shared(param_i.get_value()*0.)
        c_update = (delta_i, rho * delta_i - lrate * grad_i)
        mom_updates.append(c_update)
        p_updates.append((param_i, param_i + delta_i))
    functions['momentum'] = theano.function(train_input, cost,
            updates=mom_updates, givens={x: NNl.get_batch(training[0],
                index, batch_size, in_dims, aug)}, mode=profmode)
    functions['update'] = theano.function([], updates=p_updates,
            mode=profmode)

    print 'done'

    return functions

def train(model, data, functions, params):
    """Generic routine to perform training on the GPU using Theano-compiled
    functions and common parameters.

    This will run through a specified number of 'epochs', each consisting of
    a full pass through the training data. The epochs are broken into batches
    as normal for Stochastic Gradient Descent.

    functions: A dictionary containing all of the necessary functions for
        training. It will at least have 'momentum', 'update', and 'train_E'
        functions. 'momentum' updates the delta for each parameter, 'update'
        applies the current delta, and 'train_E' gets the current training
        cost. For supervised training, 'val_E' will usually be included
        so you can keep track of your progress on the validation set.
    params: Necessary training params: LR, training_batches, n_epochs, verbose,
        validation_batches, error (links to where best error is tracked).
    """
    LR = params['LR']

    print "Training {} epochs at LR = {} rho = {}".format(
            params['n_epochs'], LR, params['rho'])

    # If an iteration doesn't improve the validation score, we add the LR
    # to an accumulator and will reduce the LR if LR_tgt is reached
    LR_tgt = 8 * LR
    LR_cum = 0

    # reference augmentation for checking error (centered, no flip)
    T_aug = model.ref_aug

    # Main training loop
    start_time = time.clock()
    for epoch in range(params['n_epochs']):

        for chunk_i in range(len(data.b_samples)):
            data.T[0].set_value(data.raw[chunk_i])
            data.T[1].set_value(np.asarray(data.labels[chunk_i], dtype=data.ltype))

            for batch_i in range(params['t_batches'][chunk_i]):
    
                functions['momentum'](batch_i, LR, model.gen_aug())
                functions['update']()

            #if params['verb'] and (batch_i + 1) % int(Nb / 5) == 0: print '.',
            print '.',

        # check the weight distribution
        model.param_status(epoch, output=open("wlog", 'a'))

        # compute error on test and validation set
        c_train_error = [functions['train_E'](i, T_aug) for i in xrange(
                params['t_batches'][-1])]

        if epoch in params['LRsched']:
            LR = params['LRsched'][epoch]

        err_train = np.mean(c_train_error)
        if 'val_E' in functions:
            c_val_error = [functions['val_E'](i, T_aug)
                    for i in xrange(params['v_batches'])]
            err_val = np.mean(c_val_error)
        else:
            err_val = err_train

        # if we achieved a new best validation score
        # save the model and best validation score
        if err_val < getattr(params['error'], "best_error"):
            if params['verb']:
                print 'S',
            setattr(params['error'], "best_error", err_val)
            model.save_model()
#           LR_cum -= LR
#           LR_cum = max(0, LR_cum)

        else:
            print ' ',
#           LR_cum += LR
#           if LR_cum > LR_tgt:
#               LR /= 2.
#               LR_tgt /= 1.2
#               LR_cum = 0

        curr_time = NNl.nice_time(time.clock() - start_time)

        if 'val_E' in functions:
            if params['verb']:
                print("{} | epoch {: >4}, LR={:.4f}, train: {:.5f}, val: {:.5f}"
                        .format(curr_time, epoch, LR, err_train, err_val))
            else:
                print '.',
            params['logfile'].write("{} {: >4} {:.6f} {:.8f} {:.8f}\n".format(
                    curr_time, epoch, LR, err_train, err_val))
        else:
            if params['verb']:
                print("{} | epoch {: >4}, LR={:.5f}, train: {:.6f}".format(
                    curr_time, epoch, LR, err_train))
            params['logfile'].write("{} {: >4} {:.6f} {:.8f}\n".format(
                    curr_time, epoch, LR, err_train))

def class_probs(model, data, train_params):
    """ Creates Theano function to return the probabilities of class
        membership.

    Args:
        model: Model instance
        samples: set of data points to use (note: just data, not a Dataset)
        batch_size: size of the batch for calculation
        n_batches: how many batches are required to span the samples

    Returns:
        An nparray of class membership probabilities. Useful for performing
        meta-analysis/transfer learning.
    """

    index = T.lscalar('index')  # index to a [mini]batch
    aug = T.ivector('aug')      # data augmentation (jitter, flip)
    in_dims = model.x_shape[2:]

    p_func = theano.function([index, aug], model.out_layer.p_y_given_x(False),
            givens={model.x: NNl.get_batch(data.V[0], index, data.batch_n,
                in_dims, aug)})

    y = [p_func(i, model.ref_aug) for i in range(train_params['v_batches']+1)]

    return np.asarray(np.concatenate(y, axis=0))
