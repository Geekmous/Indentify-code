#!/usr/bin/python
#coding:utf-8
import httplib2 as h
import os
from urllib import urlencode
import urllib2
import urllib
import json
import requests
#import Untitled as net
from theano import tensor as T
import lasagne
import cPickle as pickle
import cv2
import theano
import numpy as np
import time
import md5

from multiprocessing import Process
import CAPYH

def getCookie():
	conn = h.Http()
	resp, content = conn.request("http://210.42.121.241/servlet/GenImg")
	
	with open(str(os.getpid()) + ".jpeg", 'w') as f:
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
	cnn = None
	cnn = CAPYH.CNN(1, 30, 120)
	cnn.setParamPath("param.txt")
	

	img = cv2.imread(str(os.getpid()) + ".jpeg", cv2.IMREAD_GRAYSCALE)
	img = [img]
	indentify_code = cnn.predict([img, ])
	#print indentify_code
	result =  checkCode(indentify_code, cookie)
	#print result
	if result == True:
		Path = './Valid'
		if not os.path.exists(Path):
			os.mkdir(Path)

		img = cv2.imread(str(os.getpid()) + ".jpeg")
		md = md5.new()
		md.update(str(time.time() + os.getpid()))
		output_path = "./t/" + str(md.hexdigest()) + ".jpg"
		cv2.imwrite(output_path, img)

		f = open("./train.txt", "a")

		f.write(output_path)
		f.write(" ")
		f.write(indentify_code)
		f.write("\n")
		f.close()
#		md = md5.new()
#		md.update(str(time.time() + os.getpid()))
#		cv2.imwrite(Path + "/" + label[p1] + "/1" + str(md.hexdigest()) + ".jpeg", img1)
#		
#		md.update(str(time.time() + os.getpid()))
#		cv2.imwrite(Path + "/" + label[p2] + "/2" + str(md.hexdigest()) + ".jpeg", img2)
#		
#		md.update(str(time.time() + os.getpid()))
#		cv2.imwrite(Path + "/" + label[p3] + "/3" + str(md.hexdigest()) + ".jpeg", img3)
#		
#		md.update(str(time.time() + os.getpid()))
#		cv2.imwrite(Path + "/" + label[p4] + "/4" + str(md.hexdigest()) + ".jpeg", img4)
		
		return True
	return False

def process():
	count = 1
	True_count = 0
	
	for i in xrange(100000):
		result = getDate()	
		if result == True:
			True_count = True_count +  1
		count += 1
		print "%s : accuracy: %f" % (os.getpid(), True_count * 1.0 / count * 100)
		

if __name__ == '__main__':
	process()


	
