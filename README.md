nnGen - neural network generator
============

Dependencies:  Theano and PyLearn2. I believe if those are working, everything here should work fine. You can find instructions for installing Theano [here](http://deeplearning.net/software/theano/install.html). PyLearn2 can be cloned with:

```
git clone git://github.com/lisa-lab/pylearn2.git
```

and then needs to be added to your `PYTHONPATH` environment variable.

## Training a network

Besides the working Python code, you need to:  prepare a formatted dataset, specify the network architecture in the file `structure`, and perhaps tune some of the training parameters.

### Dataset

I prefer to package my datasets such that I can use them as generally as possible and also track the details of what is inside, such as any pre-processing. For use with nnGen, a valid dataset is a 3-key dictionary containing:
* 1) 'info': a dictionary with a few important pieces of info
* 2) 'data': a numpy array: (sample index, [channel, y-coord,] x-coord)
* 3) 'labels': in a 1 or 2d numpy array, for classification or regression resp.
The data is best prepared as a 4-dimensional numpy array because that is compatible with the convolutional net code. However, nnGen will automatically deal with missing trivial dimensions in most cases. I include an IPython notebook that has a template for creating a usable dataset from generic data.

Keep in mind that you should probably use normalized data (a natural choice being mean and std. dev. normalization of each feature).

### Structure

When you run `Main.py`, it will look for a `structure` file in the directory (or a model checkpoint file, to resume training). `structure` contains all of the information needed to build the network, layer-by-layer. You are required to have both an Input and Output layer, and then any number of convolutional, pooling, or fully-connected layers in between. The MNIST example has two samples which demonstrate different setups. Also, in `scripts/` there is `GenStructure.py` which can be used to easily generate simple structures for batch testing.

### Training parameters

Training utilizes stochastic gradient descent (SGD) with momentum and a learning rate that will change based on a fixed schedule. One can specify the base learning rate, momentum paremeter, and batch size via command line.  Also, one can pass the seed for the RNGs, for the purpose of doing experiments or catching errors.
