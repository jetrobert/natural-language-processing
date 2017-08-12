# stephen cheng
# nips-papers 1987-2016

from gensim import corpora, models
import logging

def lsi_model(filename, topic_num):
	# load texts_stem
	texts = []
	file = open(filename, 'r+')
	for line in file:
		texts.append([x.strip() for x in line.split(',')])
	# build the bag-of-words 
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	dictionary = corpora.Dictionary(texts)

	# mapping token to id
	corpus = [dictionary.doc2bow(text) for text in texts]
	
	# tf-idf
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	#setting LDA (Latent Dirichlet allocation) Topics & building Latent Semantic Indexing model
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)
	lsi.save("lsi_model")
	lsi_name = "lsi_model"
	return lsi_name

if __name__ == '__main__':
	file_name = "texts_stem"
	topic_num = 20
	lsi_name = lsi_model(file_name, topic_num)
	print "LSI Model is built in '%s' " % lsi_name





