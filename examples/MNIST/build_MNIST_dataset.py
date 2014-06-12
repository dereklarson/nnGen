import gzip
import numpy as np
import cPickle as cp

in_file = gzip.open("mnist.pkl.gz", 'rb')
train, valid, test = cp.load(in_file)

data = np.concatenate((train[0], valid[0])).reshape(-1, 1, 28, 28)
labels = np.concatenate((train[1], valid[1]))
info = {'description': "MNIST 60k, 1x28x28 with norm [0,1)",\
        'cut': 50000, 'channels': 1}

cp.dump([info, data, labels], open("Dataset", 'wb'), 2)
