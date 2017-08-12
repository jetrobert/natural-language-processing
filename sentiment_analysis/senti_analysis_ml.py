# -*- coding: utf-8 -*-

import numpy as np
import sys
import re
import codecs
import os
import jieba
from gensim.models import word2vec
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization as BO
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def readFile(filepath):
	pathDir = os.listdir(filepath)
	child = []
	for file in pathDir:
		child.append(os.path.join('%s/%s' % (filepath, file)))
	return child

def readLines(filename):
	fopen = open(filename, 'r')
	lines = []
	for x in fopen.readlines():
		if x.strip() != '':
			lines.append(x.strip())
	fopen.close()
	return lines

def parseSent(sentence):
	seg_list = jieba.cut(sentence)
	output = ''.join(list(seg_list))
	return output

def sent2word(sentence):
	segList = jieba.cut(sentence)
	segResult = []
	for w in segList:
		segResult.append(w)
	stopwords = readLines('stop_words.txt')
	newSent = []
	for word in segResult:
		if word in stopwords:
			# print "stopword: %s" % word
			continue
		else:
			newSent.append(word)
	return newSent

def getWordVecs(wordList):
	vecs = []
	for word in wordList:
		word = word.replace('\n', '')
		try:
			vecs.append(model[word])
		except KeyError:
			continue
	return np.array(vecs, dtype = 'float')

def buildVecs(filename):
	posInput = []
	with open(filename, "rb") as txtfile:
		# print txt-file
		for lines in txtfile:
			lines = lines.split('\n')
			if lines[0] == "\r" or lines[0] == "\r\n" or lines[0] == "\r\r":
				pass
			else:
				for line in lines:            
					line = list(jieba.cut(line))
					resultList = getWordVecs(line)
					# the mean vector of sentence' vectors is used to represent it
					if len(resultList) != 0:
						resultArray = sum(np.array(resultList)) / len(resultList)
						posInput.append(resultArray)
	return posInput
	
if __name__ == '__main__':

	# load word2vec model
	#model = word2vec.Word2Vec.load_word2vec_format("corpus.model.bin", binary = True)
	model = word2vec.Word2Vec.load("corpus.model")
	# read file
	filepwd_pos = readFile("test/pos")
	filepwd_neg = readFile("test/neg")
	pos_number = 0
	neg_number = 0
	posInput = []
	negInput = []
	for pos in filepwd_pos:
		pos_buildVecs = buildVecs(pos)
		posInput.extend(pos_buildVecs)
		pos_number += 1
		if pos_number == 100:
			break
	for neg in filepwd_neg:
		neg_buildVecs = buildVecs(neg)
		negInput.extend(neg_buildVecs)
		neg_number += 1
		if neg_number == 100:
			break
	
	# use 1 for positive , 0 for negative
	y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))
	X = posInput[:]
	for neg in negInput:
		X.append(neg)
	X = np.array(X)
	
	# standardization
	X = scale(X)
	
	# Plot the PCA spectrum
	pca = PCA(n_components=400)
	pca.fit(X)
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')
	plt.savefig('senti_ana_ml_pca_spectrum.png')
	plt.show()
	X_reduced = PCA(n_components = 100).fit_transform(X)
	X_reduced_train, X_reduced_test, y_reduced_train, y_reduced_test = train_test_split(X_reduced, y, test_size=0.4, random_state=1)

	# SVM (RBF) using training data with 100 dimensions
	clf = SVC(C = 2, probability = True)
	clf.fit(X_reduced_train, y_reduced_train)
	print('SVM Test Accuracy: %.2f'% clf.score(X_reduced_test, y_reduced_test))
	pred_probas = clf.predict_proba(X_reduced_test)[:,1]
	print("SVM KS value: %f" % ks_2samp(y_reduced_test, pred_probas)[0])
	
	# plot ROC curve
	#AUC = 0.92
	#KS = 0.7
	fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas)
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr, tpr, label = 'roc_auc = %.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc = 'lower right')
	plt.savefig('senti_ana_ml_roc_curve.png')
	plt.show()
	joblib.dump(clf, "SVC.pkl")
	
	# raw data
	print("Raw test:")
	print(clf.predict(X_reduced_test))
	print("Raw value:")
	print(y_reduced_test)
	test_value = clf.predict(X_reduced_test)
	index = []
	for x in range(0,len(test_value)):
		index.append(x+1)
	test_value_1 = 0
	test_value_0 = 0
	for test_value_data in test_value:
		if test_value_data == 1:
			test_value_1 += 1
		else:
			test_value_0 += 1
	y_reduced_test_1 = 0
	y_reduced_test_0 = 0
	for y_reduced_test_data in y_reduced_test:
		if y_reduced_test_data == 1:
			y_reduced_test_1 += 1
		else:
			y_reduced_test_0 += 1
	test_value_label = 'test pos: ' + str(test_value_1) + ' neg: ' + str(test_value_0)
	y_reduced_test_label = 'value pos: ' + str(y_reduced_test_1) + ' neg: ' + str(y_reduced_test_0)
	
	# plot raw curve
	plt.plot(index, test_value,'ro',label=test_value_label)
	plt.plot(index,y_reduced_test, 'b.',label=y_reduced_test_label)
	plt.xlim([0, len(test_value)])
	plt.ylim([-2, 2])
	plt.legend(loc = 'lower right')
	plt.savefig('senti_ana_ml_raw_curve.png')
	plt.show()
	
	# MLP (Multilayer perceptron) using raw train data with 100 dimensions
	model = Sequential()
	model.add(Dense(512, input_dim = 100, init = 'uniform', activation = 'tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(loss = 'binary_crossentropy',
				  optimizer = 'adam',
				  metrics = ['accuracy'])
	model.fit(X_reduced_train, y_reduced_train, nb_epoch = 20, batch_size = 16)
	score = model.evaluate(X_reduced_test, y_reduced_test, batch_size = 16)
	print ('MLP Test accuracy: ', score[1])

	pred_probas = model.predict(X_reduced_test)
	# print "KS value: %f" % KSmetric(y_reduced_test, pred_probas)[0]

	# plot ROC curve
	# AUC = 0.91
	fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas)
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc = 'lower right')
	plt.savefig('senti_ana_ml_mlp.png')
	plt.show()

