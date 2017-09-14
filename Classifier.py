#!/usr/bin/python

import cv2
import numpy as np
import time
import os
import random
import unittest
import tensorflow as tf
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
    output_num = 4

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
        #print data.shape
        data = cv2.resize(data, dsize = (32, 128))
        data = np.reshape(data, (1, 32, 128))
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

def weight_vaiable(shape, name = None):
    init = tf.truncated_normal(shape, name = name)
    return tf.Variable(init)

def bias_vaiable(shape, name = None):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)

def conv2DLayer(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def output(x, W, b):
    x = tf.nn.dropout(x, 0.9)
    return tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W) + b))


class CNN():
    def __init__(self, channel = 3, height = 1, width = 1, Most_char = 6):
        self.output = None
        
        self.isSetParam = False
        self.channel = channel
        self.height = height
        self.width = width
        self.output_num = Most_char

        self.sess = tf.InteractiveSession()
        self.graph = tf.Graph()
        self.graph.as_default()
            #input size None 30 120 1
        with tf.name_scope("input"):
            self.input_var = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
            self.target_var = tf.placeholder(tf.float32, [None, 37 * 4])
            tf.image_summary('input', self.input_var, 10)

        with tf.name_scope("conv1"):
            self.W1 = weight_vaiable([3, 3, 1, 128])
            b1 = bias_vaiable([128])
            self.conv1 = conv2DLayer(self.input_var, self.W1) + b1
            self.conv1 = tf.nn.relu(self.conv1)
            self.max_pool1 = max_pool(self.conv1)

        with tf.name_scope("conv2"):
            W2 = weight_vaiable([3, 3, 128, 64])
            b2 = bias_vaiable([64])
            self.conv2 = conv2DLayer(self.max_pool1, W2) + b2
            self.conv2 = tf.nn.relu(self.conv2)
            self.max_pool2 = max_pool(self.conv2)

        with tf.name_scope('conv3'):
            W3 = weight_vaiable([3, 3, 64, 32])
            b3 = bias_vaiable([32])

            self.conv3 = conv2DLayer(self.max_pool2, W3) + b3
            self.conv3 = tf.nn.relu(self.conv3)
            self.max_pool3 = max_pool(self.conv3)
        #None 4 16 32
        length = 4 * 16 * 32
        with tf.name_scope("output"):
            self.reshape = tf.reshape(self.max_pool3,[-1, length])
            out_num = 1024
            h_w = weight_vaiable([length, out_num])
            h_b = bias_vaiable([out_num])

            self.hidden = tf.nn.relu(tf.matmul(self.reshape, h_w) + h_b)

            h_w1 = weight_vaiable([out_num, out_num])
            h_b1= bias_vaiable([out_num])

            self.hidden1 = tf.nn.relu(tf.matmul(self.hidden, h_w1) + h_b1)

            self.output1 = output(self.hidden1, weight_vaiable((out_num, 37)), bias_vaiable([37]))
            self.output2 = output(self.hidden1, weight_vaiable((out_num, 37)), bias_vaiable([37]))
            self.output3 = output(self.hidden1, weight_vaiable((out_num, 37)), bias_vaiable([37]))
            self.output4 = output(self.hidden1, weight_vaiable((out_num, 37)), bias_vaiable([37]))

            outputs = [self.output1, self.output2,self.output3,self.output4]
            self.output = tf.concat(1, outputs)
            tf.histogram_summary('output', self.output)

        with tf.name_scope("loss"):
            self.corss_entropy = tf.reduce_mean(-tf.reduce_sum(self.target_var[:, :37] * tf.log(self.output1), 1))
            tf.scalar_summary('train/loss', self.corss_entropy)

        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):
                correct1 = tf.equal(tf.argmax(self.output1, 1), tf.argmax(self.target_var[:,:37], 1))
                correct2 = tf.equal(tf.argmax(self.output2, 1), tf.argmax(self.target_var[:,37:37 * 2], 1))
                correct3 = tf.equal(tf.argmax(self.output3, 1), tf.argmax(self.target_var[:,37 * 2:37 * 3], 1))
                correct4 = tf.equal(tf.argmax(self.output4, 1), tf.argmax(self.target_var[:,37 * 3:37 * 4], 1))
                correct = tf.cast(correct1, tf.int32) * tf.cast(correct2, tf.int32) * tf.cast(correct3, tf.int32) * tf.cast(correct4, tf.int32)

            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
            tf.scalar_summary("accuracy", self.accuracy)

        self.op = tf.train.AdamOptimizer(1e-3).minimize(self.corss_entropy)
        self.merged = tf.merge_all_summaries()
        
        tf.initialize_all_variables().run()
        #TO-DO: how to add the graph def into the summary
        self.train_writer = tf.train.SummaryWriter("./summary", self.sess.graph_def)
        #
        



        
    def test(self, X, y):
        num, channel, height, width = X.shape
        #print X.shape
        #print num, channel, height, width
        X = np.reshape(X, newshape = (num, height, width, channel))

        #print self.sess.run(self.reshape, feed_dict = {self.input_var : X, self.target_var : y})
        #print self.sess.run(self.output1, feed_dict = {self.input_var : X, self.target_var : y})
        #print self.sess.run(self.corss_entropy, feed_dict = {self.input_var : X, self.target_var : y})
        
        saver = tf.train.Saver()
        

        for i in range(1):
            _, summary = self.sess.run([self.op, self.merged], feed_dict = {self.input_var : X, self.target_var : y})
            #if i % 10 == 0:
                #print self.sess.run(self.corss_entropy, feed_dict = {self.input_var : X, self.target_var : y})
                #print self.sess.run(self.reshape, feed_dict = {self.input_var : X, self.target_var : y})
                #print self.sess.run(self.output1, feed_dict = {self.input_var : X, self.target_var : y})
            #summary = self.sess.run(self.merged, feed_dict= { self.input_var : X, self.target_var : y})
            self.train_writer.add_summary(summary, i)
        #print self.sess.run(self.corss_entropy, feed_dict = {self.input_var : X, self.target_var : y})

        self.train_writer.close() 
        saver.save(self.sess, "./model/m.ckpt")
        return self.sess.run([self.corss_entropy, self.accuracy], feed_dict = {self.input_var : X, self.target_var : y})
        

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
		
    def NetTrain(self, X_train, y_train, X_val, y_val, learning_rate = 0.01, momentum = 0.9, iterator = 200):
        #define network

        num_epochs = iterator
 
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
                #print "inputs.shape = ", inputs.shape
                #print "targets.shape = ", targets.shape
                err, acc = self.test(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            # And a full pass over the validation data:
            #val_err = 0
            #val_acc = 0
            #val_batches = 0
            #for batch in iterate_minibatches(X_val, y_val, 32, shuffle=True):
            #    inputs, targets = batch
            #    #print "inputs.shape = ", inputs.shape
            #    #print "targets.shape = ", targets.shape
            #    err, acc = val_fn(inputs, targets)
            #    val_err += err
            #    val_acc += acc
            #    val_batches += 1
                
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err )) #'''/ train_batches'''
            print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / (train_batches + 1) * 100))
            #print("  validation loss:\t\t{:.6f}".format(val_err )) #'''/ val_batches'''
            #print("  validation accuracy:\t\t{:.2f} %".format(
            #    val_acc / (val_batches + 1) * 100))
        
	
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
    print X_train.shape
    print "Building model and compiling functions..."
    X_train = X_train
    network=CNN(1,32,128)
    network.test(X_train[:128, :, :, :], y_train[:128, :])
    network.NetTrain(X_train, y_train, X_val, y_val, iterator = 20)

    #network.saveParam("param.txt")
	#param_value = lasagne.layers.get_all_param_values(network, )

#	network.saveParam("param.txt")
	
	# After training, we compute and print the test error:

if __name__ == "__main__":
    #unittest.main()
	run()
	
	
