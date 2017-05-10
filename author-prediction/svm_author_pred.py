import numpy as np
import pandas as pd
import re
import warnings
from nltk import FreqDist
from nltk.tokenize import sent_tokenize
from os import walk
from os import path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from time import time
from syllables_en import count

warnings.filterwarnings('ignore')
SRC_DIRECTORY = path.dirname(path.realpath(__file__))
CORPUS_DIRECTORY = path.join(SRC_DIRECTORY,'./data/')

class MyFreqDist(FreqDist):
	def dises(self):
		return [item for item in self if self[item] == 2]

def extract_book_contents(text):
	outlines = []
	add_lines = False
	lines = text.split('\n')
	for line in lines:
		current_line_lower = line.lower()
		if "start of this".lower() in current_line_lower or "start of the project".lower() in current_line_lower:
			add_lines = True
			continue
		if "End of the Project Gutenberg".lower() in current_line_lower:
			break
		if add_lines:
			current_line = line.replace("\\r","").strip()
			if current_line != "":
				outlines.append(current_line)
	if len(outlines) == 0:
		print(text)
	return "\n".join(outlines)

def create_pronouns():
	return set(open(path.join(SRC_DIRECTORY,'nompronouns.txt'), 'r').read().splitlines())

def create_conjunction():
	return set(open(path.join(SRC_DIRECTORY,'coordconj.txt'), 'r').read().splitlines()).union(
		   set(open(path.join(SRC_DIRECTORY,'subordconj.txt'), 'r').read().splitlines()))

def create_stop_words():
	return set(open(path.join(SRC_DIRECTORY,'smartstop.txt'), 'r').read().splitlines())

def get_file_dir_list(dir):
	file_list = []
	dir_list = []
	for (dirpath, dirname, files) in walk(dir):
		if files:
			if len(path.split(dirpath)[1]) > 0 :
				dir_list.append(path.split(dirpath)[1])
			temp = []
			for file in files:
				if not file.startswith('.'):
					temp.append(path.abspath(path.join(dirpath,file)))
			if len(temp) > 0:
				file_list.append(temp)
	return dir_list, file_list

def load_book_features(filename, smartStopWords={}, pronSet={}, conjSet={}):
	RANGE = 25
	text = extract_book_contents(open(filename, 'r').read()).lower()
	contents = re.sub('\'s|(\r\n)|-+|["_]', ' ', text) # remove \r\n, apostrophes, and dashes
	sentenceList = sent_tokenize(contents.strip())
	cleanWords = []
	sentenceLenDist = []
	pronDist = []
	conjDist = []
	sentences = []
	totalWords = 0
	wordLenDist = []
	totalSyllables = 0
	for sentence in sentenceList:
		if sentence != ".":
			pronCount = 0
			conjCount = 0
			sentences.append(sentence)
			sentenceWords = re.findall(r"[\w']+", sentence)
			totalWords += len(sentenceWords)
			sentenceLenDist.append(len(sentenceWords))
			for word in sentenceWords:
				totalSyllables += count(word)
				wordLenDist.append(len(word))
				if word in pronSet:
					pronCount+=1
				if word in conjSet:
					conjCount+=1
				if word not in smartStopWords:
					cleanWords.append(word)
			pronDist.append(pronCount)
			conjDist.append(conjCount)

	sentenceLengthFreqDist = FreqDist(sentenceLenDist)
	sentenceLengthDist = list(map(lambda x: sentenceLengthFreqDist.freq(x), range(1, RANGE)))
	sentenceLengthDist.append(1-sum(sentenceLengthDist))

	pronounFreqDist = FreqDist(pronDist)
	pronounDist = list(map(lambda x: pronounFreqDist.freq(x), range(1, RANGE)))
	pronounDist.append(1-sum(pronounDist))

	conjunctionFreqDist = FreqDist(conjDist)
	conjunctionDist = list(map(lambda x: conjunctionFreqDist.freq(x), range(1, RANGE)))
	conjunctionDist.append(1-sum(conjunctionDist))

	wordLengthFreqDist= FreqDist(wordLenDist)
	wordLengthDist = list(map(lambda x: wordLengthFreqDist.freq(x), range(1, RANGE)))
	wordLengthDist.append(1-sum(wordLengthDist))

	avgSentenceLength = np.mean(sentenceLenDist)
	avgSyllablesPerWord = float(totalSyllables)/totalWords
	readability = float(206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord))/100

	wordsFreqDist = MyFreqDist(FreqDist(cleanWords))

	numUniqueWords = len(wordsFreqDist.keys())
	numTotalWords = len(cleanWords)

	hapax = float(len(wordsFreqDist.hapaxes()))/numUniqueWords
	dis = float(len(wordsFreqDist.dises()))/numUniqueWords
	richness = float(numUniqueWords)/numTotalWords

	result = []
	result.append(hapax)
	result.append(dis)
	result.append(richness)
	result.append(readability)
	result.extend(sentenceLengthDist)
	result.extend(wordLengthDist)
	result.extend(pronounDist)
	result.extend(conjunctionDist)
	return result, numTotalWords

def confusion(true_labels, pred_labels, categories):
	test = sorted(categories)
	result = np.zeros((len(test),len(test)))
	for true_label_idx,i in enumerate(sorted(test)):
		for pred_label_idx,j in enumerate(sorted(test)):
			for a, b in zip(true_labels, pred_labels):
				if a == i and b == j:
					result[true_label_idx][pred_label_idx] += 1
	return pd.DataFrame(result.astype(np.int32), test, test)

def get_authors(item_list, authors):
	result = []
	for item in item_list:
		result.append(authors[item])
	return result

def evaluate(confusion_matrix, categories):
	precision_list = []
	recall_list = []
	f1_list = []
	for idx, item in enumerate(sorted(categories)):
		precision = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis=0)[item]
		recall = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis = 1)[item]
		f1 = 0
		if precision + recall > 0:
			f1 = 2 * (precision * recall) / (precision + recall)
		f1_list.append(f1)
		recall_list.append(recall)
		precision_list.append(precision)
	return pd.DataFrame(np.array([precision_list,recall_list,f1_list]),index=['precision', 'recall', 'f1'],
						columns=sorted(categories))


def average_f1s(evaluation_matrix, categories):
	f1 = 0.0
	count = 0
	for idx, item in enumerate(sorted(categories)):
		if item.lower() != 'o':
			f1 += evaluation_matrix.iloc[2][idx]
			count += 1
	return f1 / count

def simple_classification_without_cross_fold_validation(x, y, estimator, scoring, categories):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4) # 30% reserved for validation
	fs = SelectPercentile(scoring, percentile=20)

	pipeline = Pipeline([('featureselector', fs), ('scaler', StandardScaler()), ('estimator', estimator)])

	pipeline = OneVsRestClassifier(pipeline)

	clfer = pipeline.fit(x_train, y_train)
	y_predict_train = clfer.predict(x_train)

	print("Accuracy on training set: %2.3f" % metrics.accuracy_score(y_train, y_predict_train))

	y_predict_test = clfer.predict(x_test)
	print("\nAccuracy on testing set: %2.3f" % metrics.accuracy_score(y_test, y_predict_test))

	confusion_matrix = confusion(get_authors(y_test,categories), get_authors(y_predict_test, categories),categories)
	print("\nConfusion Matrix:")
	print(confusion_matrix)

	print("\nResult:")
	print(evaluate(confusion_matrix,categories))

	print("\nAverage f1s:")
	print(average_f1s(evaluate(confusion_matrix,categories),categories))

def load_book_features_from_corpus(dir_list, file_list, smartStopWords={}, pronSet={}, conjSet={}):

	x = []
	y = []
	t0 = time()
	totalwords = 0
	for index, files in enumerate(file_list):
		for f in files:
			y.append(dir_list[index])
			features, numwords = load_book_features(f, smartStopWords, pronSet, conjSet)
			totalwords += numwords
			x.append(features)
	le = LabelEncoder().fit(y)
	print('Processed %d books from %d authors with %d total words in %2.3fs' % (len(x), len(dir_list), totalwords,
																				time()-t0))
	return np.array(x), np.array(le.transform(y)), le

def run_classification():
	print("Author classification started...")
	dir_list, file_list = get_file_dir_list(CORPUS_DIRECTORY)
	x, y, le = load_book_features_from_corpus(dir_list, file_list, create_stop_words(), create_pronouns(),
											  create_conjunction())
	no_samples = x.shape[0]
	no_classes = len(set(y))
	print()
	print("{no_samples} samples in {no_classes} classes".format(**locals()))
	print()
	simple_classification_without_cross_fold_validation(x, y, LinearSVC(random_state=0, tol=1e-8, penalty='l2',
																		dual=True, C=1), f_classif, dir_list)

if __name__ == '__main__':
	run_classification()