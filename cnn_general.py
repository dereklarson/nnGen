#!/home/derek/anaconda/bin/python
import os, sys, time, getopt
import cPickle as cp
import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RStreams

import NNlib as NNl
from Builder import Model

profiling = False

def runit(CPFile="", n_epochs=1000, rho=0.90, LR=0.010, channels=-1, \
        batch_size=128, seed=1000):
    ''' '''

    if profiling:
        profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
    else:
        profmode = None

    training, validation, info = NNl.shareloader("Dataset", True)

    # get some parameters from the dataset
    in_dims = info[1]
    out_dim = info[2]
    if channels == -1:
        channels = info[0]

    # create RNGs, one normal one Theano, which are passed to the Builder
    rng = np.random.RandomState(seed)
    theano_rng = MRG_RStreams(rng.randint(999999))
    rngs = [rng, theano_rng]

    # compute number of minibatches for training, validation and testing
    n_train_batches = training[0].get_value().shape[0] / batch_size
    n_valid_batches = validation[0].get_value().shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    lrate = T.fscalar('lrate')  # learning rate
    aug = T.ivector('aug')      # data augmentation (jitter, flip)
    x = T.tensor4('x')          # (batch, channels) + in_dims
    if out_dim == 1:
        y = T.ivector('y')      # integer labels for classification
        training[1] = T.cast(training[1], 'int32')
        validation[1] = T.cast(validation[1], 'int32')
    else:
        y = T.matrix('y')       # float version for regression 

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    in_shape = (batch_size, channels) + in_dims

    # load the checkpoint file if it exists, otherwise use Structure
    if os.path.isfile(CPFile):
        CNN = Model(rngs, x, in_shape, "")
    else:
        CNN = Model(rngs, x, in_shape)

    # modify our input size if we are going to use translations
    in_dims = tuple( i - CNN.jitter for i in in_dims)

    print 'Compiling Theano functions...',

    # Choose our cost and validation measures (Boolean arg refers to use of dropout)
    if out_dim == 1:
        cost = CNN.out_layer.negative_log_likelihood(y, True)
        val_error = CNN.out_layer.errors(y, False)
    else:
        cost = CNN.out_layer.mean_square_error(y, True)
        val_error = CNN.out_layer.mean_square_error(y, False)

    graph_input = [index, aug]

    # create a function to compute the mistakes that are made by the model
    train_error = theano.function(graph_input, val_error, givens={
            x: NNl.GetBatch(training[0], index, batch_size, in_dims, aug),
            y: training[1][index * batch_size: (index + 1) * batch_size]}, \
                    mode=profmode)

    val_error = theano.function(graph_input, val_error, givens={
            x: NNl.GetBatch(validation[0], index, batch_size, in_dims, aug),
            y: validation[1][index * batch_size: (index + 1) * batch_size]}, \
            mode=profmode)

    # create a list of gradients for all model parameters
    grads = T.grad(cost, CNN.params)

    train_input = [index, lrate, aug]

    #update deltas
    mom_updates = []
    p_updates = []
    for grad_i, param_i, decay_i in zip(grads, CNN.params, CNN.pdecay):
        delta_i = theano.shared(param_i.get_value()*0.)
        mom_updates.append((delta_i, rho * delta_i - lrate * decay_i * param_i - lrate * grad_i))
#       mom_updates.append((delta_i, rho * delta_i - lrate * grad_i))
        p_updates.append((param_i, param_i + delta_i))
    gain_momentum = theano.function(train_input, cost, updates=mom_updates, givens={ \
            x: NNl.GetBatch(training[0], index, batch_size, in_dims, aug),
            y: training[1][index * batch_size: (index + 1) * batch_size]}, \
            mode=profmode)
    update_model = theano.function([], updates=p_updates, mode=profmode)

    print 'done'

    ###############
    # TRAIN MODEL #
    ###############

    validation_frequency = n_train_batches
    best_validation_error = CNN.best_error
    start_time = time.clock()

    epoch = iter = lr_item = 0

    logfile = open("log_train", 'w')
#   logfile.write("#{}x{}x{} -CONV> kern/fs/pool {}{}{} -HL> {} -LOG> {}\n".format( \
#           in_dims, channels, nkerns, fsize, pool, HLout, out_dim))
#   logfile.write("#epoch   test    train   LR\n")

    print 'Training', n_epochs, 'epochs: val_fr =', validation_frequency, \
            "LR =", LR, "rho =", rho

    LR_tgt = 0.50 
    LR_cum = 0

    # reference augmentation for testing (centered, no flip)
    T_aug = [CNN.jitter / 2, CNN.jitter / 2, 1, 1]

    while (epoch < n_epochs):
        epoch += 1

        for batch_i in xrange(n_train_batches):
            iter += 1

            aug_in = NNl.GenAug(rng, CNN.jitter, [CNN.flipY, CNN.flipX])
            gain_momentum(batch_i, LR, aug_in)

            update_model()

            if (iter + 1) % int(n_train_batches / 5) == 0: print '.',

            if (iter + 1) % validation_frequency == 0:
                # compute error on test and validation set
                c_train_error = [train_error(i, T_aug) for i in xrange(n_train_batches)]
                err_train = np.mean(c_train_error)
                c_val_error = [val_error(i, T_aug) for i in xrange(n_valid_batches)]
                err_test = np.mean(c_val_error)

                # if we got the best validation score until now
                if err_test < best_validation_error:

                    CNN.save_model(err_test)
                    print 'S',
                    LR_cum -= LR
                    LR_cum = max(0, LR_cum)

                    # save best validation score and iteration number
                    best_validation_error = err_test
                else:
                    print ' ',
                    LR_cum += LR
                    if LR_cum > LR_tgt:
                        LR /= 2.
                        LR_cum = 0

                curr_time = NNl.NiceTime(time.clock() - start_time)

                print("{} | epoch {: >3}, LR={:.3f}, train: {:.5f}, test: {:.5f}, ".format( \
                        curr_time, epoch, LR, err_train, err_test))

                if profiling: 
                    profmode.print_summary()
                    sys.exit(0)

                logfile.write("{: >3} {: >5} {:.8f} {:.8f} {:.3f}\n".format( \
                        epoch, curr_time, err_train, err_test, LR))

    end_time = time.clock()
    print "Best CV: %f".format(best_validation_error)

def usage():
    print "<exec> [ -h -l <lrate> -m <momentum> <CPFile> ]"
    sys.exit()

std_opts = {'-l': 'LR', '-m': 'rho', '-s': 'seed', '-n': 'n_epochs'}

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hl:m:s:n:")
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
    print pass_in
    runit(**pass_in)
