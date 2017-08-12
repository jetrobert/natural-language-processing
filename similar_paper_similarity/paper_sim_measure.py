# stephen cheng
# nips-papers 1987-2016

import pandas as pd
from gensim import corpora, models, similarities

def sim_measure(text_name, title_num, sort_num, lsi_name):
	texts = []
	file = open(text_name, 'r+')
	for line in file:
		texts.append([x.strip() for x in line.split(',')])
	dictionary = corpora.Dictionary(texts)
	# mapping token to id
	lsi = models.LsiModel.load(lsi_name)
	corpus = [dictionary.doc2bow(text) for text in texts]
	index = similarities.MatrixSimilarity(lsi[corpus])
	# demo 
	my_content = texts[title_num]
	my_bow = dictionary.doc2bow(my_content)
	my_lsi = lsi[my_bow]
	# sort
	sims = index[my_lsi]
	sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
	return sort_sims[:sort_num]

if __name__ == '__main__':
	# instance
	file_name = 'papers.csv'
	text_name = "texts_stem"
	title_num = 6
	sort_num = 8
	lsi_name = "lsi_model"
	sort_sims = sim_measure(text_name, title_num, sort_num, lsi_name)
	column2 = 'title'
	field = [column2]
	df = pd.read_csv(file_name, skipinitialspace=True, usecols=field)
	title = df.values[title_num]   
	print "Paper Title: ", title
	for title_i, sim_r in sort_sims:
		print "Similar Papers: ", df.values[title_i]
		print "Similarity Rate: ", sim_r



