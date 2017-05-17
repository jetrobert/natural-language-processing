# encoding=utf-8
import jieba

if '__name__' == '__main__':

	seg_list = jieba.cut("�����������廪��ѧ", cut_all=True)
	print("Full Mode: " + "/ ".join(seg_list))  # ȫģʽ

	seg_list = jieba.cut("�����������廪��ѧ", cut_all=False)
	print("Default Mode: " + "/ ".join(seg_list))  # ��ȷģʽ

	seg_list = jieba.cut("�����������׺��д���")  # Ĭ���Ǿ�ȷģʽ
	print(", ".join(seg_list))

	seg_list = jieba.cut_for_search("С��˶ʿ��ҵ���й���ѧԺ�������������ձ�������ѧ����")  # ��������ģʽ
	print(", ".join(seg_list))
	
	print('/'.join(jieba.cut('����ŵ�post�н�����', HMM=False)))
	
	jieba.suggest_freq(('��', '��'), True)
	
	print('/'.join(jieba.cut('��̨�С���ȷӦ�ò��ᱻ�п�', HMM=False)))
	
	jieba.suggest_freq('̨��', True)
	
	# ���Ա�ע
	import jieba.posseg as pseg
	words = pseg.cut("�Ұ������찲��")
	for word, flag in words:
		print('%s %s' % (word, flag))
	
	# tokenize
	result = jieba.tokenize(u'���ͷ�װ��Ʒ���޹�˾')
	for tk in result:
		print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
	
	result = jieba.tokenize(u'���ͷ�װ��Ʒ���޹�˾', mode='search')
	for tk in result:
		print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
		
		