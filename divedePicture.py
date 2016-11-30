#!/usr/bin/python
import cv2 as cv
import os

def devide(Source, FileFlod):
	img = cv.imread(Source)
	index_1 = Source.rfind('/')
	index_2 = Source.index('.jpeg')
	filename = Source[index_1 + 1 : index_2]
	#print filename
	cv.imwrite(FileFlod + "/" +  filename + "zero.jpeg", img[:, 0:30, :])
	cv.imwrite(FileFlod + "/" + filename + "one.jpeg", img[:, 20:50, :])
	cv.imwrite(FileFlod + "/" + filename + "tow.jpeg", img[:, 40:70, :])
	cv.imwrite(FileFlod + "/" + filename + "three.jpeg", img[:, 70:100, :])


file_list = os.listdir("./data")
#devide("./data/100.jpeg", "./processed")
for item in file_list[1:] :
	print item
	devide("./data/" + item, "./processe")
	