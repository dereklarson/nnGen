import numpy as np
import cPickle as cp

data = cp.load(open("Train_mstd"))
labels = cp.load(open("Train_labels"))
info = {'description': "CIFAR 50k, 3x32x32 with per-pixel mean-std norm",\
        'cut': 44928, 'channels': 3}

cp.dump([info, data, labels], open("dataset", 'wb'), 2)
