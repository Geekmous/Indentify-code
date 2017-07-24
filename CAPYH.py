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
import unittest

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
    output_num = 6

    f = open("./train.txt")
    labels = []
    datas = []

    for lines in f.readlines():
        path, label = lines[:-1].split(" ")
        a = np.zeros((37 * output_num))
        for i in range(len(label)):
            a[i * 37 + maps[label[i]]] = 1
        for i in range(len(label), output_num):
            a[i * 37 + 36] = 1

        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        data = np.reshape(data, (1, 30, 120))
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
class CNN():
    def __init__(self, channel = 3, height = 1, width = 1, Most_char = 6):
        self.output = None
        self.input_var = T.dtensor4('inputs')
        self.target_var = T.dmatrix('targets')
        self.isSetParam = False
        self.channel = channel
        self.height = height
        self.width = width
        self.output_num = Most_char

    def setParamPath(self, path):
        self.ParamPath = path
        self.isSetparam = True
        
    def Train(self, X, y):
        pass

    def predict(self, X):
        self.build_cnn()

        if self.isSetParam:
            try:
                self.setParam(self.output,open(self.ParamPath))
            except Exception, e:
                print e

        predict_fn = theano.function([self.input_var, ], map(lasagne.layers.get_output, self.output))
        prediction = predict_fn(X)
        strs = []
        
        for cases in range(len(prediction[0])):
            string = ""
            for i in range(len(prediction)):
                index = np.argmax(prediction[i])
                if index >= 0 and index <= 9:
                    string += str(index)
                if index >= 10 and index < 36:
                    string += chr(ord('A') + index - 10)
            strs.append(string)
        return strs

    def saveParam(self, Path):
        with open(Path, "w") as f:
            pickle.dump(lasagne.layers.get_all_param_values(self.output), f)

    def restore(self, path):
        pass

    def build_cnn(self):
        input_var = self.input_var
        target_var = self.target_var
        output_num = self.output_num
        network = lasagne.layers.InputLayer((None, self.channel, self.height, self.width), input_var = input_var)
        network = lasagne.layers.Conv2DLayer(network, num_filters= 32, filter_size=(3, 3), pad = 1, stride = 1,nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.Constant(0.))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        network = lasagne.layers.Conv2DLayer(
                network, num_filters= 32, filter_size=(3, 3), pad = 1, stride = 1,
                nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.Constant(0.))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)

        #network = lasagne.layers.Conv2DLayer(
        #        network, num_filters= 32, filter_size=(3, 3), pad = 1, stride = 1,
        #       nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.Constant(0.))
        #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        

        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units= 1024 ,
                nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.Constant(0.))

        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        self.output = []
        for i in range(output_num):
            self.output.append(lasagne.layers.DenseLayer(
                network,
                num_units= 37,
                nonlinearity=lasagne.nonlinearities.softmax))
        



		
    def NetTrain(self, X_train, y_train, X_val, y_val, learning_rate = 0.01, momentum = 0.9, iterator = 200, output_num = 10):
        #define network
        
        self.build_cnn()
        input_var = self.input_var
        target_var = self.target_var
        output = self.output 

        if self.isSetParam:
            try:
                self.setParam(self.output,open(self.ParamPath))
            except Exception, e:
                print e

        num_epochs = iterator
        
        predict = []
        target = []
        for i in range(output_num):
            predict.append(lasagne.layers.get_output(output[i]))
            target.append( target_var[:, i * 37 : (i + 1) * 37] )

        loss = 0
        for i in range(output_num):
            loss += lasagne.objectives.categorical_crossentropy(predict[i], target[i])
        loss = loss.mean()
        train_acc = 1
        for i in range(output_num):
            train_acc *= T.eq(T.argmax(predict[i], axis = 1), T.argmax(target[i], axis = 1))
        train_acc = T.mean(train_acc, dtype = theano.config.floatX)
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(output, trainable=True)
        
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=learning_rate, momentum = momentum)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_predict = []
        test_loss = 0
        for i in range(output_num):
            test_predict.append( lasagne.layers.get_output(output[i], deterministic=True) )
            test_loss += lasagne.objectives.categorical_crossentropy(test_predict[i],target[i])

        
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = 1
        for i in range(output_num):
            test_acc *= T.eq(T.argmax(test_predict[i], axis = 1), T.argmax(target[i], axis = 1))
        test_acc = T.mean(test_acc, dtype = theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            train_acc = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 16, shuffle=True):
                inputs, targets = batch
                print "inputs.shape = ", inputs.shape
                #print "targets.shape = ", targets.shape
                err, acc  = train_fn(inputs, targets)
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
        
	
    def setParam(self, network, files):
        assert isinstance(files, file)
        param_value = pickle.load(files)
        lasagne.layers.set_all_param_values(network, param_value)


        

    #def saveParam(self, param_value, Path):
    #    with open(Path, "w") as f
    #        pickle.dump(lasagne.layers.get_all_param_values(self.output), f)
		
		
    def Predication(self, network, inputs):
        maps = {}
        index = 0
        for i in range(10):
            maps[index] = str(i)
            index += 1
        
        for i in range(10,37):
            maps[index] = chr(ord('A') + i - 10)
            index += 1

        predication0 = lasagne.layers.get_output(network[0])
        predication1 = lasagne.layers.get_output(network[1])
        predication2 = lasagne.layers.get_output(network[2])
        predication3 = lasagne.layers.get_output(network[3])
        p_function = theano.function([input_var,], [predication0,predication1,predication2,predication3])

        p = p_function(inputs)
        strs = []
        
        for i in range(4):
            chars = np.argmax(p[i], axis = 1)
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

class CNN_TEST(unittest.TestCase):
    def test_create(self):
        cnn = CNN(1, 30, 120)
        cnn.build_cnn()

    def test_train(self):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        cnn = CNN(1, 30, 120)
        cnn.NetTrain(X_train[:32, :, :, :], y_train[:32, :], X_val, y_val, iterator = 1)

        p = cnn.predict(X_train)
        strs = []
        for i in range(y_train.shape[0]):
            string = ""
            for j in range(10):
                pre = y_train[i, j * 37: (j + 1) * 37]
                t = np.argmax(pre)
                if t >= 0 and t <= 9:
                    string += str(t)
                if t >= 10 and t < 37:
                    string += chr(ord('A') + t - 10)
            strs.append(string)

        print p
    def test_predict(self):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        cnn = CNN(1, 30, 120)
        cnn.setParamPath("./param.txt")
        p = cnn.predict(X_train)

def run():
	flag_readParam = False
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	print(X_train.shape)

	print("Building model and compiling functions...")
	
	network = CNN()
	if flag_readParam:
		network.setParamPath("param.txt")
    

	network.NetTrain(X_train, y_train, X_val, y_val, iterator = 1)

    #network.saveParam("param.txt")
	#param_value = lasagne.layers.get_all_param_values(network, )

	network.saveParam(param_value, "param.txt")
	
	# After training, we compute and print the test error:

if __name__ == "__main__":
    unittest.main()
	#run()
	
	