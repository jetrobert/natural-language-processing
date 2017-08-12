from gensim.models import Word2Vec

if __name__ == "__main__":
	en_wiki_word2vec_model = Word2Vec.load('enwiki_word2vec_gensim')
	en_wiki_word2vec_model.most_similar('word')
	en_wiki_word2vec_model.most_similar('happy')
	en_wiki_word2vec_model.most_similar(positive=['woman', 'queen'], negative=['man'], topn=8)
	en_wiki_word2vec_model.similarity('woman', 'man')
	en_wiki_word2vec_model.doesnt_match("breakfast cereal dinner lunch".split())