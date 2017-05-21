# Digit Classification on SVHN Dataset using Keras Sequential Interface

1) Task 1
* Run a multilayer perceptron two hidden layers and rectified linear nonlinearities on the iris dataset.

2) Task 2
* Train a multilayer perceptron on the MNIST dataset
* Compare a “vanilla” model with a model Qusing drop-out
* Visualize the learning curves

3) Task 3
* Train a convolutional neural network on the SVHN dataset in format 2 (single digit classification)
* Build a model using batch normalization

4) Task 4
* Load the weights of a pre-trained convolutional neural network (AlexNet or VGG)
* Use it as feature extraction method to train a linear model or MLP  on the pets dataset.

5) Performance:

* Task 1:
	* Test Accuracy: 0.86842105263157898
* Task 2:
	* Vanila Model:
		* Validation Accuracy: 0.972167
	* Model after Dropout:
		* Validation Accuracy: 0.983000
* Task 3:
	* Base model Test Accuracy: 0.866
	* Model after Batch Normalization Test Accuracy: 0.84

* Task 4:
	* LinearVC:
		* Test Accuracy: 0.78841991342
	* LogisticRegression:
		* Test Accuracy: 0.802489177489
