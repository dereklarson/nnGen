This example shows a simple network training on the MNIST handwritten digits dataset. The included "Structure" file details an ANN with 2 fully-connected layers with ReLU units and a final classification layer using a rectified and scaled tanh function. 

###Data preparation

In order to use the code, you need to prepare the dataset to the format I established.  This can be done easily with the provided scripts:  

* `build_MNIST_dataset.py` -- for command line, or
* `Dataset Builder (MNIST).ipynb` -- IPython notebook for interactivity

Both of these will create a file "dataset" which should be in the folder when you run "Main.py", and both assume you have the dataset provided for the Theano tutorials. Explicitly, for the first sample you could run the following,

$ ./Main.py -d dataset -S sample1_struc

though the filename 'dataset' will automatically be checked.

You can find the dataset here:
	
	http://deeplearning.net/data/mnist/mnist.pkl.gz

###Comparison

The included `sample1_log` is an example of output during a training run which achieves < 1.4% validation error in under 2 minutes on a GeForce GTX 660, with no data augmentation, just dropout.

You should compare to Geoff Hinton's group's results:

	http://arxiv.org/pdf/1207.0580.pdf

Then, 'sample2_log' is a run with a convolutional net using data augmentation and achieves ~0.4% error in 10 minutes. A list of academic results on popular datasets can be found here for comparison:

	http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
