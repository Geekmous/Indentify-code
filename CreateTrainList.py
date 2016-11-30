#!/usr/bin/python

import os

label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C', 'D','E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

Path = "./Train"
if not os.path.exists(Path):
	os.mkdir(Path)
f = open("./Train.txt", "w")

for index in xrange(len(label)):
	i = label[index]
	file_list = os.listdir(Path + "/" + i)
	#print file_list
	for file_name in file_list[1:]:
		f.write(Path + "/" + i + "/" + file_name + " " + str(index))
		f.write("\n")
		
f.close()
Path = "./Valid"
if not os.path.exists(Path):
	os.mkdir(Path)
	
f = open("./Valid.txt", "w")
for index in xrange(len(label)):
	i = label[index]
	file_list = os.listdir(Path + "/" + i)
	#print file_list
	for file_name in file_list[1:]:
		f.write(Path + "/" + i + "/" + file_name + " " + str(index))
		f.write("\n")
		
f.close()
