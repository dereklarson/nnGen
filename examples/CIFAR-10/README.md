This example shows a convolutional neural network (CNN) training on the CIFAR-10 dataset. The included structure file produces a network similar to one used by Alex Krizhevsky in his cuda-convnet project.

###Data preparation

In order to use the code, you need to prepare the dataset to the format I established.  I started with the dataset found on Kaggle:

	http://www.kaggle.com/c/cifar-10/data

Download the training data and labels into your working directory and extract the training images, which should end up in `./images_train/`.  Then use the provided scripts:  

`PrepData.py` -- this produces a pickled numpy array of the pixel data

`PrepLabels.py` -- this produces a pickled numpy array of the labels

`build_CIFAR10_dataset.py` -- combines the data, labels with some info to make a "Dataset"

`nnTrain.py` will look for `Dataset` when running, as well as "Structure".

###Comparison
	
The included `sample_log_train` is an example of output during a training run which achieves < 21% test error in under 20 minutes on a GeForce GTX 660.

Here is Alex K.'s cuda-convnet page, compare to the 26% "fast results" example:

	https://code.google.com/p/cuda-convnet/

