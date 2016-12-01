#!/usr/bin/python


import os

labels = os.listdir("./test")
for label in labels[1:]:
	files = os.listdir("./test/" + label)
	for file in files[1:]:
		if os.path.exists("./test/" + label + "/" + file):
			os.remove("./test/" + label + "/" + file)
		