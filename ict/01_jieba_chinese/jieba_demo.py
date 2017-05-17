# encoding=utf-8
import jieba

if '__name__' == '__main__':

	seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
	print("Full Mode: " + "/ ".join(seg_list))  # 全模式

	seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
	print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

	seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
	print(", ".join(seg_list))

	seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
	print(", ".join(seg_list))
	
	print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
	
	jieba.suggest_freq(('中', '将'), True)
	
	print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
	
	jieba.suggest_freq('台中', True)
	
	# 词性标注
	import jieba.posseg as pseg
	words = pseg.cut("我爱北京天安门")
	for word, flag in words:
		print('%s %s' % (word, flag))
	
	# tokenize
	result = jieba.tokenize(u'永和服装饰品有限公司')
	for tk in result:
		print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
	
	result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
	for tk in result:
		print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
		
		