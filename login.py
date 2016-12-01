#!/usr/bin/python
#coding:utf-8
import httplib2 as h
import os
from urllib import urlencode
import urllib2
import urllib
import json
import requests
import Untitled as net
from theano import tensor as T
import lasagne
import cPickle as pickle
import cv2
import theano
import numpy as np
import time
import md5
def getCookie():
	conn = h.Http()
	resp, content = conn.request("http://210.42.121.241/servlet/GenImg")
	with open("indentify.jpeg", 'w') as f:
		f.write(content)
	return resp['set-cookie']
	
def checkCode(Indentify_code, cookie):
	url = "http://210.42.121.132/servlet/Login"
	
	data = {'id' : "", 'pwd' : "", "xdvfb" : Indentify_code}
	data = 'id=&pwd=&xdvfb=' + str(Indentify_code)
	
	head = {'Cookie' : cookie,
	'Connection' : 'keep-alive',
	'Content-Type': "application/x-www-form-urlencoded",
	"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:50.0) Gecko/20100101 Firefox/50.0",
	"Accept-Encoding": "gzip, deflate",
	"Referer" :"http://210.42.121.132/",
	"Host" : "21042.121.132",
	"Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	"Upgrade-Insecure_Requests" : "1"
	}
	
	r = requests.post(url, data = data, headers = head, allow_redirects=False)
	print r.status_code
	assert r.status_code == 302 or r.status_code == 200
	if r.status_code == 302:
		return True
	else:
		return False
def getDate():
	cookie = getCookie()
	input_var = T.ftensor4()
	network = net.build_cnn(input_var)
	param_value = None
	with open("param.txt", 'r') as f:
		param_value = pickle.load(f)
	lasagne.layers.set_all_param_values(network, param_value)
	output = lasagne.layers.get_output(network)
	prediction = theano.function([input_var,], output, allow_input_downcast=True)

	img = cv2.imread("indentify.jpeg", cv2.IMREAD_GRAYSCALE)
	print img.shape
	img1 = img[:, 0:30]
	img2 = img[:, 20:50]
	img3 = img[:, 40:70]
	img4 = img[:, 70:100]

	i1 = img1.reshape((1, 1, 30, 30))
	i2 = img2.reshape((1, 1, 30, 30))
	i3 = img3.reshape((1, 1, 30, 30))
	i4 = img4.reshape((1, 1, 30, 30))
	i1 = (i1 - 128.0) / 128.0
	i2 = (i2 - 128.0) / 128.0
	i3 = (i3 - 128.0) / 128.0
	i4 = (i4 - 128.0) / 128.0

	p1 = prediction(i1)
	p2 = prediction(i2)
	p3 = prediction(i3)
	p4 = prediction(i4)


	p1 = np.argmax(p1, axis = 1)
	p2 = np.argmax(p2, axis = 1)
	p3 = np.argmax(p3, axis = 1)
	p4 = np.argmax(p4, axis = 1)

	label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C', 'D','E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

	indentify_code = label[p1] + label[p2] + label[p3] + label[p4]
	print indentify_code
	#indentify_code = raw_input("indentify:")
	#print type(indentify_code)
	result =  checkCode(indentify_code, cookie)
	print result
	if result == True:
		Path = './Train'
		if not os.path.exists(Path):
			os.mkdir(Path)
		if not os.path.exists(Path + "/" + label[p1]):
			os.mkdir(Path + "/" + label[p1])
		if not os.path.exists(Path + "/" + label[p2]):
			os.mkdir(Path + "/" + label[p2])
		if not os.path.exists(Path + "/" + label[p3]):
			os.mkdir(Path + "/" + label[p3])
		if not os.path.exists(Path + "/" + label[p4]):
			os.mkdir(Path + "/" + label[p4])
		md = md5.new()
		md.update(str(time.time()))
		cv2.imwrite(Path + "/" + label[p1] + "/" + str(md.hexdigest()) + ".jpeg", img1)
		
		md.update(str(time.time()))
		cv2.imwrite(Path + "/" + label[p2] + "/" + str(md.hexdigest()) + ".jpeg", img2)
		
		md.update(str(time.time()))
		cv2.imwrite(Path + "/" + label[p3] + "/" + str(md.hexdigest()) + ".jpeg", img3)
		
		md.update(str(time.time()))
		cv2.imwrite(Path + "/" + label[p4] + "/" + str(md.hexdigest()) + ".jpeg", img4)
for i in xrange(100000):
	result = getDate()	
	print result
	with open("result.txt", 'w') as f:
		f.write(str(result) + '\n')