# -*- coding:utf-8-*-
#coding=gbk

import os
import jieba

# read file in lib
def eachFile(filepath):
	pathDir = sorted(os.listdir(filepath))
	child = []
	for allDir in pathDir:
		child.append(os.path.join('%s/%s' % (filepath, allDir)))
	return child

# read lines in filename
def read2Lines(filename):
	file = open(filename, 'r')
	lines=[]
	for x in file.readlines():
		if x.strip() != '' or x.strip() != None:
			lines.append(x.strip())
	file.close()
	return lines
	
# split words
def sent2word(sentence):
	segList = jieba.cut(sentence)
	segResult = []
	for w in segList:
		segResult.append(w)
	stopwords = read2Lines('stop_words.txt')
	newSent = []
	for word in segResult:
		if word + '\n' in stopwords:
			continue
		else:
			newSent.append(word)
	return newSent
	
filepwd = eachFile("test/neg")
score_list = []
i = 0
for file in filepwd:
	data = read2Lines(file)
	#data_gbk = data[0].decode('gbk')
	word = sent2word(data[0])
	if word == []:
		continue
	print word

