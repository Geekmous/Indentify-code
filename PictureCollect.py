#!/usr/bin/python

import httplib2
import re
import md5
import time
import os
Path = "./data/"
if not os.path.exists(Path):
	os.mkdir(Path)
	
md = md5.new()

for i in xrange(10000):
	h = httplib2.Http()
	resp, content = h.request("http://210.42.121.133/servlet/GenImg?rdddd=1")
	md.update(str(time.time()))
	with open(str(Path + str(md.hexdigest()) + ".jpeg"), "w") as f:
		f.write(content)
	print i
