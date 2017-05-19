import jieba  
import re  
import codecs  
import os 
from gensim.models import word2vec
import logging

def readLines(filename):
	# read txt or csv file
	fp = open(filename, 'r')
	lines = []
	for line in fp.readlines():
		line = line.strip()
		line = line.decode("utf-8")
		lines.append(line)
	fp.close()
	return lines

def parseSent(sentence):
	# use Jieba to parse sentences
	seg_list = jieba.cut(sentence)
	output = ' '.join(list(seg_list)) # use space to join them
	return output
	
def getWordVecs(wordList):
	vecs = []
	for word in wordList:
		word = word.replace('\n', '')
		try:
			# only use the first 500 dimensions as input dimension
			vecs.append(model[word])
		except KeyError:
			continue
	# vecs = np.concatenate(vecs)
	return np.array(vecs, dtype = 'float')
	
if __name__ == '__main__':
	# only content is valid
	pattern = "<content>(.*?)</content>"
	csvfile = codecs.open("corpus.csv", 'w', 'utf-8')
	fileDir = os.listdir("./SogouCA/")
	for file in fileDir:
		with open("./SogouCA/%s" % file, "r") as txtfile:
			for line in txtfile:
				m = re.match(pattern, line)
				if m:
					segSent = parseSent(m.group(1))
					csvfile.write("%s" % segSent)
					
	logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
	sentences = word2vec.Text8Corpus("corpus.csv")  
	model = word2vec.Word2Vec(sentences, size = 400)

	# save model
	model.save("corpus.model")
	# model = word2vec.Word2Vec.load("corpus.model")
	
	# save model with C language methoed
	model.wv.save_word2vec_format("corpus.model.bin", binary = True)
	# model = word2vec.Word2Vec.load_word2vec_format("corpus.model.bin", binary=True)
					
					