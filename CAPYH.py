#!/usr/bin/python

import cv2
import numpy as np
import lasagne
import theano
from theano import tensor as T
import time
import os
import cPickle as pickle
import random

input_var = T.dtensor4('inputs')
target_var = T.dmatrix('targets')
#return num * width * height * g 
#return y = num * str_name
def getData():
    maps = {}
    index = 0
    for i in range(10):
        maps[str(i)] = index
        index += 1
    
    for i in range(26):
        maps[chr(ord('A') + i)] = index
        index += 1

    Path = "./t"
    lists = os.listdir(Path)
    labels = []
    datas = []
    for name in lists:
        if name.find(".jpg") >= 0:
            label = name[:-4]
            a = np.zeros((36 * 4))
            for i in range(4):
                a[i * 36 + maps[label[i]]] = 1
            data = cv2.imread(os.path.join(Path, name))
            data = np.reshape(data, (3, 30, 120))
            data = data / 256.0
            labels.append(a)
            datas.append(data)

	Xtrain = []
    ytrain = []
    Xvalid = []
    yvalid = []
    ratio = 0.2
    for i in range(len(labels)):
        if random.random() > ratio:
            Xtrain.append(datas[i])
            ytrain.append(labels[i])
        else:
            Xvalid.append(datas[i])
            yvalid.append(labels[i])
    
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xvalid = np.array(Xvalid)
    yvalid = np.array(yvalid)
    return Xtrain, ytrain, Xvalid, yvalid
	
def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	
	X_train, y_train, X_val, y_val = getData()
	#print y_train
	#print np.argmax(y_train, axis = 0)
	
	X_test = None
	y_test = None
	
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn():
    network = lasagne.layers.InputLayer((None, 3, 30, 120), input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(3, 3), pad = 1, stride = 1,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
    network = lasagne.layers.Conv2DLayer(
			network, num_filters=64, filter_size=(3, 3), pad = 1, stride = 1,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)

    network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(3, 3), pad = 1, stride = 1,
			nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
	

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units= 3072 ,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
    net1 = lasagne.layers.DenseLayer(
			network,
			num_units= 36,
			nonlinearity=lasagne.nonlinearities.softmax)
    net2 = lasagne.layers.DenseLayer(
			network,
			num_units= 36,
			nonlinearity=lasagne.nonlinearities.softmax)
    net3 = lasagne.layers.DenseLayer(
			network,
			num_units= 36,
			nonlinearity=lasagne.nonlinearities.softmax)
    net4 = lasagne.layers.DenseLayer(
			network,
			num_units= 36,
			nonlinearity=lasagne.nonlinearities.softmax)
    return (net1, net2, net3, net4)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]
		
def NetTrain(network, X_train, y_train, X_val, y_val, learning_rate = 0.01, momentum = 0.9, iterator = 200):
    num_epochs = iterator
	
    pre0 = lasagne.layers.get_output(network[0])
    pre1 = lasagne.layers.get_output(network[1])
    pre2 = lasagne.layers.get_output(network[2])
    pre3 = lasagne.layers.get_output(network[3])
    
    target0 = target_var[:, 0 * 36:(0 + 1) * 36]
    target1 = target_var[:, 1 * 36:(1 + 1) * 36]
    target2 = target_var[:, 2 * 36:(2 + 1) * 36]
    target3 = target_var[:, 3 * 36:(3 + 1) * 36]
    loss0 = lasagne.objectives.categorical_crossentropy(pre0, target0)
    loss1 = lasagne.objectives.categorical_crossentropy(pre1, target1)
    loss2 = lasagne.objectives.categorical_crossentropy(pre2, target2)
    loss3 = lasagne.objectives.categorical_crossentropy(pre3, target3)

    loss = (loss0 + loss1 + loss2 + loss3).mean()
    train_acc = T.eq(pre0, target0)
    train_acc += T.eq(pre1, target1)
    train_acc += T.eq(pre2, target2)
    train_acc += T.eq(pre3, target3)
    train_acc = T.mean(train_acc) / 4.0
    
    #train_acc = T.mean(T.eq(prediction[:, 0:36], target_var[:, 0:36]), dtype=theano.config.floatX)
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
	
    updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=learning_rate, momentum = momentum)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
    test_prediction1 = lasagne.layers.get_output(network[0], deterministic=True)
    test_prediction2 = lasagne.layers.get_output(network[1], deterministic=True)
    test_prediction3 = lasagne.layers.get_output(network[2], deterministic=True)
    test_prediction4 = lasagne.layers.get_output(network[3], deterministic=True)
    test_loss1 = lasagne.objectives.categorical_crossentropy(test_prediction1,target0)
    test_loss2 = lasagne.objectives.categorical_crossentropy(test_prediction2,target1)
    test_loss3 = lasagne.objectives.categorical_crossentropy(test_prediction3,target2)
    test_loss4 = lasagne.objectives.categorical_crossentropy(test_prediction4,target3)
    
    
    test_loss = (test_loss1 + test_loss2 + test_loss3 + test_loss4).mean()
	# As a bonus, also create an expression for the classification accuracy:
    test_acc = T.eq(test_prediction1, target0)
    test_acc += T.eq(test_prediction2, target1)
    test_acc += T.eq(test_prediction3, target2)
    test_acc += T.eq(test_prediction4, target3)

    test_acc = T.mean(test_acc, dtype = theano.config.floatX)
 
	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates,)

	# Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], )
	
    print("Starting training...")
	# We iterate over epochs:
    for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
    	train_err = 0
    	train_batches = 0
    	train_acc = 0
    	start_time = time.time()
    	for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True):
            inputs, targets = batch
            #print "inputs.shape = ", inputs.shape
            #print "targets.shape = ", targets.shape
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1

		# And a full pass over the validation data:
    	val_err = 0
    	val_acc = 0
    	val_batches = 0
    	for batch in iterate_minibatches(X_val, y_val, 32, shuffle=True):
            inputs, targets = batch
            #print "inputs.shape = ", inputs.shape
            #print "targets.shape = ", targets.shape
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
			
		
    	print("Epoch {} of {} took {:.3f}s".format(
    		epoch + 1, num_epochs, time.time() - start_time))
    	print("  training loss:\t\t{:.6f}".format(train_err )) #'''/ train_batches'''
    	print("  training accuracy:\t\t{:.2f} %".format(
    	train_acc / (train_batches + 1) * 100))
    	print("  validation loss:\t\t{:.6f}".format(val_err )) #'''/ val_batches'''
    	print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / (val_batches + 1) * 100))
	
    return network
	
def setParam(network, files):
	assert isinstance(files, file)
	param_value = pickle.load(files)
	lasagne.layers.set_all_param_values(network, param_value)
	
def saveParam(param_value, Path):
	
	with open(Path, 'w') as f:
		pickle.dump(param_value, f)
		
		
def Predication(network, inputs):
    maps = {}
    index = 0
    for i in range(10):
        maps[index] = str(i)
        index += 1
    
    for i in range(10,37):
        maps[index] = chr(ord('A') + i - 10)
        index += 1

    predication = lasagne.layers.get_output(network)
    pre = theano.function([input_var,], predication)
    p = pre(inputs)
    p = np.array(p)
    strs = []
    print np.argmax(p[:, 0 * 36:(0 + 1) * 36])
    for i in range(4):
        chars = np.argmax(p[:, i * 36:(i + 1) * 36], axis = 1)
        strs.append(chars)
    result = []
    length = len(strs[0])
    for i in range(length):
        s = ""
        for j in range(4):
            s += maps[strs[j][i]]
        result.append(s)

    return result
	
# Load the dataset
def run():
	flag_readParam = False
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	print(X_train.shape)

	print("Building model and compiling functions...")
	
	network = build_cnn()
	if flag_readParam:
		with open("param.txt", 'r') as f:
			setParam(network, f)
    
	network = NetTrain(network, X_train, y_train, X_val, y_val, iterator = 10)

	param_value = lasagne.layers.get_all_param_values(network, )

	saveParam(param_value, "param.txt")
	

	# After training, we compute and print the test error:

if __name__ == "__main__":
	run()
	
	