# stephen cheng
# nips-papers 1987-2016

import pandas as pd
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
import logging
import os.path
import sys

def sim_measure(filename, column, topic_num, title_num, sort_num):
	# read csv file
	fields = [column]
	df = pd.read_csv(filename, skipinitialspace=True, usecols=fields)
	contents = []
	for value in df.values:
		contents.append(value[0])

	# split words
	texts_lower = [[word for word in document.lower().split()] for document in contents[:40]]
	texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in contents[:40]]

	# filter stop words & symbols
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

	# build the bag-of-words 
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	dictionary = corpora.Dictionary(texts)

	# mapping tokens to indices
	corpus = [dictionary.doc2bow(text) for text in texts]
	
	# tf-idf
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	#setting LDA (Latent Dirichlet allocation) Topics & building Latent Semantic Indexing model
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)
	lsi.save("lsi_model_demo")
	index = similarities.MatrixSimilarity(lsi[corpus])

	# demo                         
	my_content = texts[title_num]
	my_bow = dictionary.doc2bow(my_content)
	lsi = models.LsiModel.load("lsi_model_demo")
	my_lsi = lsi[my_bow]
	
	# sort
	sims = index[my_lsi]
	sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
	return sort_sims[:sort_num]

if __name__ == '__main__':
	filename = 'papers.csv'
	column, column2 = 'paper_text', 'title'
	# instance
	topic_num = 20
	title_num = 8
	sort_num = 8
	sort_sims = sim_measure(filename, column, topic_num, title_num, sort_num)
	field = [column2]
	df = pd.read_csv(filename, skipinitialspace=True, usecols=field)
	title = df.values[title_num]   
	print "Paper Title: ", title
	for title_i, sim_r in sort_sims:
		print "Similar Papers: ", df.values[title_i]
		print "Similarity Rate: ", sim_r





