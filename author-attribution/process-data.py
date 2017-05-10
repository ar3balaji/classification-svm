import requests
import re
import os
import string
from bs4 import BeautifulSoup

remove = ["Produced by","End of the Project Gutenberg","End of Project Gutenberg"]

def process_downloaded_books():
	print("Preprocessing of text started...")
	sourcepattern = re.compile(".*\.txt$")
	sourceDir = "./data"
	outputDir = "./data-processed"
	try:
		print("Creating output directory...")
		os.makedirs(outputDir)
	except:
		print("Output Directory already exit...continue processing...")
	for fn in os.listdir(sourceDir):
		if sourcepattern.match(fn):
			beautify(sourceDir+'/'+fn, outputDir+'/', fn)

def beautify(fn, outputDir, filename):
	with open(fn, 'r') as input_file:
		lines = input_file.readlines()
	lines = lines[0].split('\\n')
	author = ""
	outlines = []
	add_lines = False
	for line in lines:
		if line.startswith("Author: "):
			author = line[8:]
			continue
		if "chapter 1" in line.lower() or "chapter i" in line.lower():
			add_lines = True
			continue
		if "End of the Project Gutenberg" in line:
			break
		if add_lines:
			current_line = line.replace("\\r","").strip()
			if current_line != "":
				outlines.append(current_line)
	ofn = filename
	if author != "":
		ofn = ofn.replace(".txt", "_".join(author.lower().replace("\\r","").split(" "))+".txt")
		text = "\n".join(outlines)
		if len(outlines) > 2500:
			text = "\n".join(outlines[:2500])
		if text != "":
			with open(outputDir+ofn, "w+") as f:
				f.write(text)

def dowload_top_100_books():
	print("download starting...")
	base_url = 'http://www.gutenberg.org/'
	response = requests.get('http://www.gutenberg.org/browse/scores/top')
	soup = BeautifulSoup(response.text)
	h_tag = soup.find(id='books-last30')
	ol_tag = h_tag.next_sibling.next_sibling
	file_count = 0
	output_directory = './data'
	try:
		print("Creating output directory...")
		os.makedirs(output_directory)
	except:
		print("Output Directory already exit...continue processing...")
	for a_tag in ol_tag.find_all('a'):
		m = re.match(r'(.*)(\(\d+\))', a_tag.text)
		m = re.match(r'/ebooks/(\d+)', a_tag.get('href'))
		book_id = m.group(1)
		url = base_url + '/'.join(list(book_id[:-1])) + '/' + book_id + '/' + book_id + '.txt'
		r = requests.get(url)
		if r.status_code == requests.codes.ok:
			file = output_directory + '/Chapter'+str(file_count)+'.txt'
			with open(file, 'w+') as f:
				f.write(str(r.text.encode('UTF-8')))
		else:
			print('Failed for ', book_id)
		file_count +=1
	print("Download of files done...")

def download_books():
	base_url = 'http://www.gutenberg.org'
	author_start_name = list(string.ascii_lowercase)
	response = requests.get('http://www.gutenberg.org/browse/authors/a')
	soup = BeautifulSoup(response.text)
	author_div = soup.findAll("div", { "class" : "pgdbbyauthor" })

if __name__ == '__main__':
	process_downloaded_books()