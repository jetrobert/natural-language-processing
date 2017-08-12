# stephen cheng
# nips-papers 1987-2016

import pandas as pd
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import logging
import os.path
import sys

def read_data(column, filename):
	# read csv file
	fields = [column]
	df = pd.read_csv(filename, skipinitialspace=True, usecols=fields)
	contents = []
	for value in df.values:
		contents.append(value[0])
	return contents

def extract_word(contents):
	# split words
	interval = len(contents) / 500  # == 13
	for i in range(interval+1):
		in_i, in_j = i * 500, (i + 1) * 500
		if in_i == interval * 500:
			texts_lower = [[word for word in document.lower().split()] for document in contents[in_i:]]
			texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in contents[in_i:]]
		else:
			texts_lower = [[word for word in document.lower().split()] for document in contents[in_i:in_j]]
			texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in contents[in_i:in_j]]

		# filter stopwords & symbols
		english_stopwords = stopwords.words('english')
		texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
		english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '+', '_','--', \
			'=', '/', '>', '<', '~', '...', '\\', '__', '=~', '{', '}']
		texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]

		# stemming
		st = LancasterStemmer()
		texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]

		# dropping word-frequency == 1
		all_stems = sum(texts_stemmed, [])
		stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
		texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
		program = os.path.basename(sys.argv[0])
		logger = logging.getLogger(program)
		i = 0
		file = open('texts_stem', 'a')
		for text in texts:
			file.write(','.join(text) + "\n")
			i = i + 1
			if (i % 100 == 0):
				logger.info("Saved " + str(i) + " papers")
		file.close()
		file_name = 'texts_stem'
	return file_name

if __name__ == '__main__':
	filename = 'papers.csv'
	column = 'paper_text'
	data = read_data(column, filename)
	file_name = extract_word(data)
	print "Words extracted saved in '%s' " % file_name





