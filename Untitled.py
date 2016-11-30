#!/usr/bin/python

import cv2
import numpy as np
import lasagne
import theano
from theano import tensor as T
import time
import os
import cPickle as pickle
def getData(table):
	ratio = 3
	count = 0
	output_num = 36
	Xtrain = None
	ytrain = None
	Xvalid = None
	yvalid = None
	with open('./Train.txt', 'r') as input:
		lines = input.readlines()
		for line in lines:
			path, label = line.split(' ')
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			img = img.reshape((1, img.shape[0] * img.shape[1]))
			img = (img - 128.0) / 128.0
			#print img.shape
			flag = int(label)
			label = np.zeros((1, output_num), dtype = 'int32')
			label[0, flag] = 1
			#print label
		
			if Xtrain == None:
				Xtrain = img
			else:
				Xtrain = np.concatenate((Xtrain, img), axis = 0)
				
			if ytrain == None:
						ytrain = label
			else:
				ytrain = np.concatenate((ytrain, label), axis = 0)
			
	with open('./Valid.txt', 'r') as input:
		lines = input.readlines()
		for line in lines:
			path, label = line.split(' ')
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			img = img.reshape((1, img.shape[0] * img.shape[1]))
			img = (img - 128.0) / 128.0
			#print img.shape
			flag = int(label)
			label = np.zeros((1, output_num), dtype = 'int32')
			label[0, flag] = 1
			#print label
		
			if Xvalid == None:
				Xvalid = img
			else:
				Xvalid = np.concatenate((Xvalid, img), axis = 0)
					
			if yvalid == None:
				yvalid= label
			else:
				yvalid = np.concatenate((yvalid, label), axis = 0)
	return Xtrain, ytrain.T, Xvalid, yvalid.T
	
def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	
	X_train, y_train, X_val, y_val = getData('Train.txt')
	#print y_train
	#print np.argmax(y_train, axis = 0)
	
	X_train = X_train.reshape( (X_train.shape[0], 1, 30, 30))
	y_train = np.argmax(y_train, axis = 0)
	
	
	X_val = X_val.reshape( (X_val.shape[0], 1, 30, 30))
	y_val = np.argmax(y_val, axis = 0)
	
	X_test = None
	y_test = None
	
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn(input_var=None):
	
	network = lasagne.layers.InputLayer((None, 1, 30, 30), input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(3, 3), pad = 1,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(3, 3), pad = 1,
			nonlinearity=lasagne.nonlinearities.rectify)
			
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride = 2)

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.8),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.8),
			num_units=36,
			nonlinearity=lasagne.nonlinearities.softmax)

	return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]
		
def NetTrain(network ,X_train, y_train, X_val, y_val):
	num_epochs = 50
# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	
	target_var = T.lvector('targets')
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	
	train_acc = T.mean(T.eq(T.argmax(prediction, axis = 1), target_var), dtype=theano.config.floatX)
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates,)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc], )
	
	t_p = theano.function([input_var,], test_prediction, )
	
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		train_acc = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			err, acc = train_fn(inputs, targets)
			train_err += err
			train_acc += acc
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 200, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1
			p = t_p(inputs,)
			#print (p)
			#print (np.argmax( p,axis = 1 ))
			#print (targets)

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err )) #'''/ train_batches'''
		print("  training accuracy:\t\t{:.2f} %".format(
		train_acc / train_batches * 100))
		print("  validation loss:\t\t{:.6f}".format(val_err )) #'''/ val_batches'''
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))
	
	return network
		
# Load the dataset

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

print(X_train.shape)

# Prepare Theano variables for inputs and targets

# Create neural network model (depending on first command line parameter)

print("Building model and compiling functions...")

input_var = T.dtensor4('inputs')

network = build_cnn(input_var)

with open("param.txt", 'r') as f:
	param_value = pickle.load(f)

lasagne.layers.set_all_param_values(network, param_value, )

network = NetTrain(network, X_train, y_train, X_val, y_val)

param_value = lasagne.layers.get_all_param_values(network, )

with open("param.txt", 'w') as f:
	pickle.dump(param_value, f)
	
predication = lasagne.layers.get_output(network)
pre = theano.function([input_var,], predication,)

# After training, we compute and print the test error:
	
'''
file_list = os.listdir('./processe')
x = None
img_list = []
path_list = []
for path in file_list[1:1000]:
	img = cv2.imread('./processe/' + path, cv2.IMREAD_GRAYSCALE)
	i = cv2.imread('./processe/' + path, cv2.IMREAD_GRAYSCALE)
	i = i.reshape((1, 1, 30, 30))
	i = (i - 128.0) / 128.0
	if x == None:
			x = i
	else:
		x = np.concatenate((x, i), axis = 0)
		
	img_list.append(img)
	path_list.append(path)
p = pre(x)
p = np.argmax(p, axis = 1)
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C', 'D','E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print "p : ", p
for i in xrange(len(p)):
	print "predication: %s and file_name : %s" % (label[p[i]], path_list[i])
	cv2.imwrite('./test/' + label[p[i]] + '/' + path_list[i], img_list[i])

'''
	
	
	
'''	
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
	inputs, targets = batch
	err, acc = val_fn(inputs, targets)
	test_err += err
	test_acc += acc
	test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
	test_acc / test_batches * 100))
def getData(table):
	ratio = 5
	count = 0
	output_num = 36
	Xtrain = None
	ytrain = None
	Xvalid = None
	yvalid = None
	with open(table, 'r') as input:
		
		lines = input.readlines()
		for line in lines:
			path, label = line.split(' ')
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			img = img.reshape((1, img.shape[0] * img.shape[1]))
			img = (img - np.mean(img)) / np.std(img)
			#print img.shape
			flag = int(label)
			label = np.zeros((1, output_num))
			label[0, flag] = 1
			if count == ratio:
				if Xvalid == None:
					Xvalid = img
				else:
					Xvalid = np.concatenate((Xvalid, img), axis = 0)
					
				if yvalid == None:
					yvalid= label
				else:
					yvalid = np.concatenate((yvalid, label), axis = 0)
				count = 0
			else :
				if Xtrain == None:
					Xtrain = img
				else:
					Xtrain = np.concatenate((Xtrain, img), axis = 0)
				if ytrain == None:
						ytrain = label
				else:
					ytrain = np.concatenate((ytrain, label), axis = 0)
				count += 1
				
	return Xtrain, ytrain.T, Xvalid, yvalid.T

def ReLU(x):
	x[x < 0.0] = 0.0
	return x

def Der_ReLU(x):
	d = np.ones(x.shape)
	d[d < 0] = 0
	return d
	
def sigmoid(x):
	t = np.exp(-x)
	t[t < 1e-6] = 1e-6
	return 1.0 / (1.0 + t)
def Der_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
	
def error(predicate, label):
	t = (predicate - label)
	return 1.0 / 2.0 * t.dot(t.T)
	
def net(Xtrain, W, ytrain,hidden_layer1_num, hidden_layer2_num, output_num):
	m, n = Xtrain.shape
	delta = 1
	W1 = W[0, 0 : n * hidden_layer1_num].reshape((n, hidden_layer1_num))
	W2 = W[0, n * hidden_layer1_num : n * hidden_layer1_num + hidden_layer1_num * hidden_layer2_num].reshape(( hidden_layer1_num, hidden_layer2_num))
	W3 = W[0, n * hidden_layer1_num + hidden_layer1_num * hidden_layer2_num : ].reshape(( hidden_layer2_num, output_num))
	grad_all = 0;
	errors = 0;
	for i in xrange(m):
		x = Xtrain[i ,0:].reshape((1, n))
		#print "x : ", x
		y = ytrain[0:, i].reshape((1, output_num))
		#print ""y
		h1 = x.dot(W1)
		#print "h1 :", h1
		z1 = sigmoid(h1)
		#print "z1 :", z1
		h2 = z1.dot(W2)
		#print "h2 :", h2
		z2 = sigmoid(h2)
		#print "z2 :", z2
		h3 = z2.dot(W3)
		#print "h3 :", h3
		output = sigmoid(h3)
		#print "output: ", output
	
		error3 = (y - output) * Der_sigmoid(h3)
	
		grad3 = h2.T.dot(error3)
		#print "grad3 : ", grad3
		
		error2 = Der_sigmoid(h2) * (error3.dot(W3.T))

		grad2 = z1.T.dot(error2)
		#print "grad2 : ", grad2
		error1 = Der_sigmoid(h1) * (error2.dot(W2.T))
	
		grad1 = x.T.dot(error1)
		#print "grad1 : ", grad1
		grad = np.concatenate((grad3.reshape((1, grad3.shape[0] * grad3.shape[1])), grad2.reshape((1, grad2.shape[0] * grad2.shape[1])), grad1.reshape((1, grad1.shape[0] * grad1.shape[1]))), axis = 1)
		
		grad_all += delta * grad
		#print "grad_all : ", grad_all
		errors += error(output, y)
	return output, errors, grad_all

def predicate(x, W, hidden_layer1_num, hidden_layer2_num, output_num):
	m, n = x.shape
	W1 = W[0, 0 : n * hidden_layer1_num].reshape((n, hidden_layer1_num))
	W2 = W[0, n * hidden_layer1_num : n * hidden_layer1_num + hidden_layer1_num * hidden_layer2_num].reshape(( hidden_layer1_num, hidden_layer2_num))
	W3 = W[0, n * hidden_layer1_num + hidden_layer1_num * hidden_layer2_num : ].reshape(( hidden_layer2_num, output_num))
	
	#print ""y
	h1 = x.dot(W1)
	#print "h1 :", h1
	z1 = sigmoid(h1)
	#print "z1 :", z1
	h2 = z1.dot(W2)
	#print "h2 :", h2
	z2 = sigmoid(h2)
	#print "z2 :", z2
	h3 = z2.dot(W3)
	#print "h3 :", h3
	output = sigmoid(h3)
	
	
	return np.argmax(output)
	
def coss_valid(Xvalid, W, yvalid, hidden_layer1_num, hidden_layer2_num, output_num):
	m, n = Xvalid.shape
	accuracy = 0;
	count = 0;
	for i in xrange(m):
		x = Xvalid[i, :].reshape( (1, n))
		y = yvalid[0:, i].reshape( (1, output_num))
		pre = predicate(x, W, hidden_layer1_num, hidden_layer2_num, output_num)
		
		if yvalid[pre, i] == 1:
			count += 1
			
	accuracy = 1.0 * count / n
	
	return accuracy
	
	
	
	
	
Xtrain, ytrain, Xvalid, yvalid = getData("Train.txt")

hidden_layer1_num = 500
hidden_layer2_num = 250
output_num = 36

W = np.zeros((1, Xtrain.shape[1] * hidden_layer1_num + hidden_layer1_num * hidden_layer2_num + hidden_layer2_num * output_num)) + 1e-5
output = 0
for intertor in xrange(10000):
	output, loss, grad = net(Xtrain, W, ytrain, hidden_layer1_num, hidden_layer2_num, output_num)
	#print output
	#print loss
	#print output
	W = W + grad
	#print "W", W
	print output
	print "loss : ", loss
	accuracy = coss_valid(Xvalid, W, yvalid, hidden_layer1_num, hidden_layer2_num, output_num)
	print "accuracy : ", accuracy
	'''
