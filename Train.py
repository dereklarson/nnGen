import time
import numpy as np

import theano
import theano.tensor as T

import NNlib as NNl                 #Helper library with useful functions

def create_functions(model, training, validation, batch_size, rho, profmode):

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
    graph_input = [index, aug]

    functions['train_E'] = theano.function(graph_input, model.val_error, givens={
            x: NNl.get_batch(training[0], index, batch_size, in_dims, aug),
            y: NNl.get_batch_0(training[1], index, batch_size)}, mode=profmode)

    functions['val_E'] = theano.function(graph_input, model.val_error, givens={
            x: NNl.get_batch(validation[0], index, batch_size, in_dims, aug),
            y: NNl.get_batch_0(validation[1], index, batch_size)}, mode=profmode)

    # create a list of gradients for all model parameters
    grads = T.grad(model.cost, model.params)

    train_input = [index, lrate, aug]

    # Functions to update the model, via momentum-based SGD
    mom_updates = []
    p_updates = []
    for grad_i, param_i in zip(grads, model.params):
        delta_i = theano.shared(param_i.get_value()*0.)
        c_update = (delta_i, rho * delta_i - lrate * grad_i)
        mom_updates.append(c_update)
        p_updates.append((param_i, param_i + delta_i))
    functions['momentum'] = theano.function(train_input, model.cost, updates=mom_updates, givens={ \
            x: NNl.get_batch(training[0], index, batch_size, in_dims, aug),
            y: NNl.get_batch_0(training[1], index, batch_size)}, mode=profmode)
    functions['update'] = theano.function([], updates=p_updates, mode=profmode)

    print 'done'

    return functions

def create_functions_ae(layer, training, activation, batch_size, rho, x, in_dims, profmode):

    print 'Compiling autoencoder functions...', 

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    lrate = T.fscalar('lrate')  # learning rate
    aug = T.ivector('aug')      # data augmentation (jitter, flip)

    # Functions to calculate our training and validation error while we run
    functions = {}

    cost = layer.reconstruct_mse(activation)

    graph_input = [index, aug]

    functions['train_E'] = theano.function(graph_input, cost, givens={
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
    functions['momentum'] = theano.function(train_input, cost, updates=mom_updates, givens={ \
            x: NNl.get_batch(training[0], index, batch_size, in_dims, aug)}, mode=profmode)
    functions['update'] = theano.function([], updates=p_updates, mode=profmode)

    print 'done'

    return functions

def train(model, functions, params):
    LR = params['LR']
    Nb = params['t_batches']

    print 'Training', params['n_epochs'], 'epochs at LR =', LR

    # If an iteration doesn't improve the validation score, we add the LR
    # to an accumulator and will reduce the LR if LR_tgt is reached
    LR_tgt = 8 * LR
    LR_cum = 0

    # reference augmentation for checking error (centered, no flip)
    T_aug = [model.jitterX / 2, model.jitterY / 2, 1, 1]

    # Main training loop
    start_time = time.clock()
    for epoch in range(params['n_epochs']):

        for batch_i in range(Nb):

            functions['momentum'](batch_i, LR, model.gen_aug())
            functions['update']()

            if (batch_i + 1) % int(Nb / 5) == 0: print '.',

        # compute error on test and validation set
        c_train_error = [functions['train_E'](i, T_aug) for i in xrange(Nb)]

        err_train = np.mean(c_train_error)
        if 'val_E' in functions:
            c_val_error = [functions['val_E'](i, T_aug) for i in xrange(params['v_batches'])]
            err_val = np.mean(c_val_error)
        else:
            err_val = err_train

        # if we achieved a new best validation score
        # save the model and best validation score 
        if err_val < getattr(params['error'], "best_error"):
            print 'S',
            setattr(params['error'], "best_error", err_val)
            LR_cum -= LR
            LR_cum = max(0, LR_cum)

        else:
            print ' ',
            LR_cum += LR
            if LR_cum > LR_tgt:
                LR /= 2.
                LR_tgt /= 1.2
                LR_cum = 0

        curr_time = NNl.nice_time(time.clock() - start_time)

        if 'val_E' in functions:
            print("{} | epoch {: >4}, LR={:.4f}, train: {:.5f}, val: {:0.5f}".format( \
                    curr_time, epoch, LR, err_train, err_val))
            params['logfile'].write("{} {: >4} {:.4f} {:.8f} {:.8f}\n".format( \
                    curr_time, epoch, LR, err_train, err_val))
        else:
            print("{} | epoch {: >4}, LR={:.5f}, train: {:.6f}".format( \
                    curr_time, epoch, LR, err_train))
            params['logfile'].write("{} {: >4} {:.4f} {:.8f}\n".format( \
                    curr_time, epoch, LR, err_train))
