#!/usr/bin/python

import httplib2
import re
import md5
import time
Path = "./data/"

md = md5.new()

for i in xrange(10000):
	h = httplib2.Http()
	resp, content = h.request("http://210.42.121.133/servlet/GenImg?rdddd=1")
	md.update(str(time.time()))
	with open(str(Path + str(md.hexdigest()) + ".jpeg"), "w") as f:
		f.write(content)
	print i
