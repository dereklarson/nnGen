This example shows a simple network training on the MNIST handwritten digits dataset. The included "Structure" file details an ANN with 2 fully-connected layers with ReLU units and a final classification layer using a rectified and scaled tanh function. 

###Data preparation

In order to use the code, you need to prepare the dataset to the format I established.  This can be done easily with the provided scripts:  

"build_MNIST_dataset.py" -- for command line, or
"Dataset Builder (MNIST).ipynb" -- IPython notebook for interactivity

Both of these will create a file "Dataset" which should be in the folder when you run "nnTrain.py", and both assume you have the dataset provided for the Theano tutorials.  

You can find the dataset here:
	
	http://deeplearning.net/data/mnist/mnist.pkl.gz

###Comparison

The included "sample_log_train" is an example of output during a training run which achieves < 1.4% test error in under 2 minutes on a GeForce GTX 660.

You should also compare to Geoff Hinton's group's results:

	http://arxiv.org/pdf/1207.0580.pdf
