# -*- coding:utf-8 -*-

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import math

def read_file(filename):
	of = open(filename, 'r+')
	text = []
	for line in of:
		text.append(line.split()[0])
	of.close()
	return text

def load_text(filename):
	openf = open(filename, 'r')
	t = ''
	for i in openf:
		t += i.strip()
	line_list = t.split("\n")
	text = ' '.join(line_list)
	#print text.decode('utf-8')
	openf.close()
	return text

def text_process(file_1, file_2, stopwords_name, symbols_name):
	text_1, text_2 = load_text(file_1), load_text(file_2)
	contents = [text_1, text_2]

	# split words 
	texts_cut = [[word for word in (" ".join(jieba.cut(document, cut_all=False))).encode('utf-8').split()] for document in contents]

	# filter stop words & symbols
	en_stopwords = read_file(stopwords_name)
	texts_filtered_stopwords = [[word for word in document if not word in en_stopwords] for document in texts_cut]
	english_punctuations = read_file(symbols_name)
	texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
	
	return texts_filtered
	
# tf_idf function, return tfidf array
def tf_idf(word_list):
	text_list = []
	for text in word_list:
		texts = ' '.join(text)
		text_list.append(texts)
	vectorList = text_list
	
	vectorizer = CountVectorizer()
	ft = vectorizer.fit_transform(vectorList)
	counts = ft.toarray()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(counts)	
	tfidf_ = tfidf.toarray()
	
	return tfidf_
	
def cos_value(a, b):
	tfidf_ = tf_idf(word_list)
	xy, X_, Y_ = 0, 0, 0
	for x, y in zip(tfidf_[a], tfidf_[b]):
		xy += x * y
		X_ += x**2
		Y_ += y**2
		
	return xy / (math.sqrt(X_) * math.sqrt(Y_)) 

if __name__ == '__main__':
	paper_1, paper_2 = '1.txt', '2.txt'
	stopwords_name = 'stopwords_zh.txt'
	symbols_name = 'symbols_en.txt'
	word_list = text_process(paper_1, paper_2, stopwords_name, symbols_name)
	a, b = 0, 1
	cos_sim = cos_value(a, b)
	print 'Cosine Similarity: ', cos_sim
	
	