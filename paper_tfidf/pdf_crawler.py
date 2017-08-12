import urllib2
import re
import os

# open the url and read
def getHtml(url):
	page = urllib2.urlopen(url)
	html = page.read()
	page.close()
	return html

# compile the regular expressions and find
def getUrl(html):
	reg = r'(?:href|HREF)="?((?:http://)?.+?\.pdf)'
	url_re = re.compile(reg)
	url_lst = re.findall(url_re,html)
	return url_lst

def getFile(url):
	file_name = 'paper_pdf/' + url.split('/')[-1]
	u = urllib2.urlopen(url)
	f = open(file_name, 'wb')

	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break
		f.write(buffer)
	f.close()
	print "Successfully downloaded" + " " + file_name

if __name__ == '__main__':
	
	root_url = 'https://arxiv.org/pdf/1611.001'
	#raw_url = 'https://arxiv.org/pdf/1611.00123.pdf'
	#html = getHtml(raw_url)
	#url_lst = getUrl(root_url)

	for i in range(22,99):
		temp = str(i) + '.pdf'
		url = root_url + temp
		getFile(url)