This example shows a convolutional neural network (CNN) training on the CIFAR-10 dataset. The included structure file produces a network similar to one used by Alex Krizhevsky in his cuda-convnet project.

###Data preparation

In order to use the code, you need to prepare the dataset to the format I established.  I started with the dataset found on Kaggle:

	http://www.kaggle.com/c/cifar-10/data

Download the training data and labels into your working directory and extract the training images, which should end up in `./images_train/`.  Then use the provided scripts:  

* `PrepData.py` -- this produces a pickled numpy array of the pixel data
* `PrepLabels.py` -- this produces a pickled numpy array of the labels
* `build_CIFAR10_dataset.py` -- combines the data, labels with some info to make a "Dataset"

`Main.py` will look for `dataset` when running, as well as "structure", so either rename the latter or you can pass it in as:

$ ./Main.py -S sample1_struc

###Comparison
	
The included `sample1_log` is an example of output during a training run which achieves ~ 13% test error in around 200 epochs, 3 hours, on a GeForce GTX 660.

Here is Alex K.'s cuda-convnet page, compare to the 13% example:

	https://code.google.com/p/cuda-convnet/

A few differences: 
* Alex's used a training/validation/test split, with multistaged training
* Alex's GPU setup runs ~2-3 faster
