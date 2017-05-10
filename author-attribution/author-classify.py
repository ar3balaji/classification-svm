import numpy as np
import nltk, os
import random, re
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
from os import listdir
from os import path, stat
from os.path import isfile, join

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
chapters = []
all_text = ""

def beautify(fn, outputDir, filename):
	with open(fn, 'r') as input_file:
		lines = input_file.readlines()
	author = ""
	outlines = []
	add_lines = False
	for line in lines:
		current_line_lower = line.lower()
		if "Author:" in line:
			author = line[8:]
			continue
		if "start of this".lower() in current_line_lower or "start of the project".lower() in current_line_lower:
			add_lines = True
			continue
		if "End of the Project Gutenberg".lower() in current_line_lower:
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
			with open(os.path.abspath(outputDir)+'/'+ofn, "w+") as f:
				f.write(text)

def process_downloaded_books():
	print("Preprocessing of text started...")
	sourcepattern = re.compile(".*\.txt$")
	sourceDir = os.path.abspath("../author-prediction/data")
	outputDir = "./data-processed"
	try:
		print("Creating output directory...")
		os.makedirs(outputDir)
	except:
		print("Output Directory already exit...continue processing...")
	for fn in os.listdir(sourceDir):
		if not fn.startswith('.'):
			for file in os.listdir(os.path.abspath(sourceDir+'/'+ fn)):
				if sourcepattern.match(file):
					beautify(os.path.abspath(sourceDir+'/'+fn+'/'+file), os.path.abspath(outputDir), file)

def get_file_author(fn):
	result = re.sub(r'text[0-9]*', '', fn)
	result = result.replace(".txt", "").strip()
	return result

def evaluate_model():
	classification_result=[]
	data_folder = r"./data-processed"
	test_file_pairs = []
	files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and not f.startswith('.')]

	for i in range(len(files)-2+1):
		temp = random.randrange(0,len(files)-1)
		temp1 = random.randrange(0,len(files)-1)
		test_file_pairs.append((files[temp], files[temp1]))

	# add random same file pairs
	if len(files) >= 5:
		for i in range(5):
			temp = random.randrange(0,len(files)-1)
			test_file_pairs.append((files[temp],files[temp]))

	for file_pair in test_file_pairs:
		print()
		print("processing pair...")
		print(file_pair)
		error = False
		if (stat(path.join(data_folder,file_pair[0])).st_size == 0):
			print("File does not have data so skipping it...")
			continue
		if (stat(path.join(data_folder,file_pair[1])).st_size == 0):
			print("File does not have data so skipping it...")
			continue
		if get_file_author(file_pair[0]) == get_file_author(file_pair[1]):
			set_global_data([path.join(data_folder,file_pair[0]),join(data_folder,file_pair[1])])
			try:
				result = get_cluster_labels()
			except:
				print("Runtime error occured.."+ "skip this file pair...")
				error = True
			if error:
				continue
			if result[0] == result[1]:
				classification_result.append(1)
			else:
				classification_result.append(0)
		else:
			set_global_data([path.join(data_folder,file_pair[0]),join(data_folder,file_pair[1])])
			try:
				result = get_cluster_labels()
			except:
				print("Runtime error occured.."+ "skip this file pair...")
				error = True
			if error:
				continue
			if result[0] == result[1]:
				classification_result.append(0)
			else:
				classification_result.append(1)
	print("Processing completed...")
	return classification_result

def set_global_data(files):
	global chapters
	chapters = []
	global all_text
	all_text = ""
	for fn in files:
		with open(fn) as f:
			chapters.append(f.read().replace('\n', ' '))
	all_text = ' '.join(chapters)

def cluster_documents(feature_vectors):
	km = KMeans(n_clusters=2, init='k-means++', n_init=10, verbose=0)
	km.fit(feature_vectors)
	return km

def get_punctuation_features():
	punctuation = np.zeros((len(chapters), 3), np.float64)
	for e, ch_text in enumerate(chapters):
		tokens = nltk.word_tokenize(ch_text.lower())
		total_sentences = len(sentence_tokenizer.tokenize(ch_text))
		punctuation[e, 0] = tokens.count(',') / float(total_sentences)
		punctuation[e, 1] = tokens.count(';') / float(total_sentences)
		punctuation[e, 2] = tokens.count(':') / float(total_sentences)
	return whiten(punctuation)

def get_lexical_features():
	fvs_lexical = np.zeros((len(chapters), 3), np.float64)
	for e, ch_text in enumerate(chapters):
		words = word_tokenizer.tokenize(ch_text.lower())
		sentences = sentence_tokenizer.tokenize(ch_text)
		vocab = set(words)
		words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
									   for s in sentences])
		fvs_lexical[e, 0] = words_per_sentence.mean()
		fvs_lexical[e, 1] = words_per_sentence.std()
		fvs_lexical[e, 2] = len(vocab) / float(len(words))
	fvs_lexical = whiten(fvs_lexical)
	return fvs_lexical

def get_bag_of_words():
	# get most common words in the whole book
	NUM_TOP_WORDS = 10
	all_tokens = nltk.word_tokenize(all_text)
	fdist = nltk.FreqDist(all_tokens)
	vocab = list(fdist.keys())[:NUM_TOP_WORDS]
	vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
	fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)
	fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]
	return fvs_bow

def get_syntactic_features():
	def token_to_pos(ch):
		tokens = nltk.word_tokenize(ch)
		return [p[1] for p in pos_tag(tokens)]
	chapters_pos = [token_to_pos(ch) for ch in chapters]
	pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
	fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
						   for ch in chapters_pos]).astype(np.float64)
	fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
	return fvs_syntax

def get_cluster_labels():
	result = []
	return_result =[]
	result_dict = {0:{0:0,1:0},1:{0:0,1:0}}
	# for lexical features result
	result.append(cluster_documents(get_lexical_features()).labels_)

	# for lexical punctuation features result
	result.append(cluster_documents(get_punctuation_features()).labels_)

	# for bag of words result
	result.append(cluster_documents(get_bag_of_words()).labels_)

	# for syntactic features result
	result.append(cluster_documents(get_syntactic_features()).labels_)

	for i, item in enumerate(result):
		for j, chapter in enumerate(item):
			result_dict[j][chapter] += 1

	if result_dict[0][0] > result_dict[0][1]:
		return_result.append(0)
	else:
		return_result.append(1)

	if result_dict[1][0] > result_dict[1][1]:
		return_result.append(0)
	else:
		return_result.append(1)

	return return_result

if __name__ == '__main__':
	# o-classified incorrectly 1-classified correctly
	process_downloaded_books()
	result = evaluate_model()
	print()
	print("Evaluation results:")
	print("\tNumber of file pairs classified correctly: "+str(result.count(1)))
	print("\tNumber of file pairs Incorrectly classified: "+str(result.count(0)))
	print("\tAccurate percentage: " + str(round((result.count(1)/len(result)) * 100,2)) + '%')
	print()
