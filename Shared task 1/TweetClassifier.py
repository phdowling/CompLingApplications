import string
from nltk.text import TextCollection
from nltk.stem.porter import PorterStemmer
import nltk

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

STOPWORDS = ['a', 'about', 'accord', 'across', 'after', 'afterward', 'again', "ain't", 'all', 'almost', 'alon', 'along', 'alreadi', 'also', 'alway', 'am', 'among', 'amongst', 'an', 'and', 'anoth', 'ani', 'anybodi', 'anyon', 'anyth', 'anywher', 'apart', 'are', "aren't", 'around', 'as', 'asid', 'ask', 'ask', 'associ', 'at', 'avail', 'away', 'be', 'becam', 'becaus', 'becom', 'becom', 'becom', 'been', 'befor', 'beforehand', 'behind', 'be', 'below', 'besid', 'between', 'beyond', 'both', 'brief', 'but', 'by', "c'mon", 'c', 'came', 'can', "can't", 'cannot', 'cant', 'caus', 'caus', 'chang', 'co', 'com', 'contain', 'contain', 'contain', 'correspond', "couldn't", 'cours', 'current', 'describ', 'did', "didn't", 'do', 'doe', "doesn't", 'do', "don't", 'done', 'downward', 'dure', 'each', 'edu', 'eg', 'eight', 'either', 'elsewher', 'et', 'etc', 'ever', 'everi', 'everybodi', 'everyon', 'everyth', 'everywher', 'ex', 'exampl', 'far', 'few', 'fifth', 'first', 'five', 'follow', 'follow', 'follow', 'for', 'former', 'former', 'forth', 'four', 'from', 'get', 'get', 'get', 'given', 'give', 'go', 'goe', 'go', 'gone', 'got', 'gotten', 'had', "hadn't", 'happen', 'has', "hasn't", 'have', "haven't", 'have', 'he', 'he', 'hello', 'henc', 'her', 'here', 'here', 'hereaft', 'herebi', 'herein', 'hereupon', 'her', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'how', 'howbeit', "i'd", "i'll", "i'm", "i'v", 'ie', 'if', 'in', 'inasmuch', 'inc', 'indic', 'indic', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", 'it', 'it', 'itself', 'keep', 'keep', 'kept', 'known', 'last', 'late', 'later', 'latter', 'latter', 'lest', 'let', 'let', 'like', 'look', 'ltd', 'main', 'mani', 'may', 'me', 'meanwhil', 'more', 'my', 'myself', 'name', 'name', 'nd', 'near', 'near', 'neither', 'never', 'new', 'next', 'nine', 'no', 'nobodi', 'non', 'none', 'noon', 'nor', 'not', 'noth', 'now', 'nowher', 'of', 'off', 'often', 'ok', 'old', 'on', 'onc', 'one', 'one', 'onli', 'onto', 'or', 'other', 'other', 'otherwis', 'our', 'our', 'ourselv', 'out', 'over', 'overal', 'own', 'per', 'place', 'plus', 'probabl', 'que', 'qv', 'rd', 're', 'regard', 'said', 'same', 'saw', 'say', 'say', 'say', 'second', 'second', 'see', 'see', 'seem', 'seem', 'seen', 'self', 'selv', 'sent', 'seven', 'shall', 'she', "shouldn't", 'sinc', 'six', 'some', 'somebodi', 'somehow', 'someon', 'someth', 'sometim', 'sometim', 'somewhat', 'somewher', 'soon', 'specifi', 'specifi', 'specifi', 'sub', 'sup', 't', 'take', 'taken', 'tell', 'tend', 'th', 'than', 'thanx', 'that', 'that', 'that', 'the', 'their', 'their', 'them', 'themselv', 'then', 'thenc', 'there', 'there', 'thereaft', 'therebi', 'therein', 'there', 'thereupon', 'these', 'they', "they'd", "they'll", "they'r", "they'v", 'third', 'this', 'those', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'took', 'toward', 'toward', 'tri', 'tri', 'tri', 'twice', 'two', 'un', 'under', 'unless', 'until', 'unto', 'up', 'upon', 'us', 'use', 'use', 'use', 'usual', 'various', 'via', 'viz', 'vs', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we'r", "we'v", 'went', 'were', "weren't", 'what', 'what', 'when', 'whenc', 'whenev', 'where', 'where', 'whereaft', 'wherea', 'wherebi', 'wherein', 'whereupon', 'wherev', 'whether', 'which', 'while', 'whither', 'who', 'who', 'whoever', 'whole', 'whom', 'whose', 'whi', 'with', 'within', 'without', "won't", "wouldn't", 'yes', 'yet', 'you', "you'd", "you'll", "you'r", "you'v", 'your', 'your', 'yourself', 'yourselv', 'zero']



"""
USAGE:
run a python command line in the same directory as this

import TweetClassifier
jc = TweetClassifier.JointClassifier(TweetClassifier.dictfile, TweetClassifier.trainfile, TweetClassifier.datafile, "outJoint.pred")
jc.train()
jc.classifyTweets()

To run the scorer, in another command line run:

python scorer/scorer.py b dev_test_set.pred out3.pred 

(replace out3 by out2 for just the NB classifier or out1 for just the weighted classifier)


"""



class Tweet(object):
	def __init__(self, id1, id2, sentiment, tweet):
		self.id1 = str(id1)
		self.id2 = str(id2)
		self.sentiment = sentiment
		self.tweet = tweet
	def toOut(self):
		return "" + self.id1 + "\t" + self.id2 + "\t" + self.sentiment + "\t" + self.tweet + "\n"

class TweetClassifier(object):
	"""
	Parent class for all classifiers. Provides some methods to read the training data, the evaluation data, and writing the results to file.
	
	"""
	def __init__(self, trainfile=None, datafile=None, outfile=None):
		self.trainfile = trainfile
		self.datafile = datafile
		self.outfile = outfile
		self.trainingTweets = []
		self.evalTweets = []

	def readDataset(self, datafile):
		with open((datafile or self.datafile),"r") as data:
			for line in data.readlines():
				fields = line.split("\t")
				tweet = Tweet(fields[0], fields[1], "unknown",fields[3])
				#print tweet.__dict__
				self.evalTweets.append(tweet)

	def readTrainingData(self, trainfile):
		with open(trainfile, "r") as tweetdata:
			for line in tweetdata.readlines():
				fields = line.split("\t")
				#print fields
				tweet = Tweet(fields[0], fields[1], fields[2], fields[3])
				self.trainingTweets.append(tweet)

	def writeResults(self, outfile):
		with open((outfile or self.outfile),"w") as out:
			for tweet in self.evalTweets:
				out.write(tweet.toOut())

	def train(self, trainfile):
		raise Exception("Call in subclass")

	def classifyTweets(self, datafile, outfile):
		raise Exception("Call in subclass")

class WeightedTweetClassifier(TweetClassifier):
	"""
	Basic idea:
	train TF-IDF model on training data
	filter out all words that we do not have clues for
	multiply all remaining term weights with the corresponding clues (+1, -1, 0), and sum the results
	"""
	def __init__(self, dictfile=None, trainfile=None, datafile=None, outfile=None):
		# Call the superclass constructor
		super(WeightedTweetClassifier, self).__init__(trainfile, datafile, outfile)
		self.stemmer = PorterStemmer()

		self.trainfile = trainfile
		self.datafile = datafile
		self.outfile = outfile

		#this contains the clues we were given: {"clue":1.0, "clue2":-1.0 ... }
		self.clueValues = {}

		#the NLTK TextCollection class is used because it provides TF-IDF functionality.
		self.textCollection = None

		# read the clues
		self.readDictionary(dictfile)

		# for saving sentiment scores, so they can be meaningfully used later on by e.g. the Joint Classifier
		self.scores = {}

	def readDictionary(self, dictfile=None):
		"""
		read the dictionary file. +1, -1 or 0 is saved as a sentiment for each (stemmed) term in self.clueValues

		TODO: maybe we don't want to stem, but instead use the provided POS tags? could be a separate classifier though
		"""
		with open(dictfile, "r") as dictdata:
			for line in dictdata.readlines():
				fields = line.split(" ")
				token = self.stemmer.stem(fields[2].split("=")[1].strip())
				polarity = fields[5].split("=")[1].strip()
				self.clueValues[token] = (1.0 if polarity == "positive" else (-1.0 if polarity == "negative" else 0.0))

	def train(self, trainfile=None):
		print "training WeightedTweetClassifier"
		self.readTrainingData((trainfile or self.trainfile))
		for tweet in self.trainingTweets:
			# lowercase, remove punctuation
			nopunct = string.lower(tweet.tweet.translate(string.maketrans("",""), string.punctuation))
			tweet.tweet = nopunct
		# add all Tweets to our TextCollection. This automatically creates a TF-IDF model
		self.textCollection = TextCollection([tweet.tweet for tweet in self.trainingTweets])

	def classifyTweets(self, datafile=None, outfile=None):
		print "reading dataset"
		self.readDataset(datafile)

		print "classifying Tweets with weighted classifier"
		for tweet in self.evalTweets:
			# score = sum of TF-IDF weighted terms which carry sentiment
			tokens = string.lower(tweet.tweet.translate(string.maketrans("",""), string.punctuation)).split(" ")
			score = sum([self.textCollection.tf_idf(token, tweet.tweet) * self.clueValues.get(self.stemmer.stem(token), 0) 
				for token in tokens])
			self.scores[(tweet.id1, tweet.id2)] = score

			# Any score very close or equal to 0 is judged to be neutral.
			tweet.sentiment = ("neutral" if abs(score) < 0.01 else ( "negative" if score < 0 else "positive"))

		self.writeResults(outfile)



class SimpleDict(object):
	"""
	Utility class to provide a mapping from token to ID. This is used to convert documents to bag-of-words format.
	The Naive Bayes classifier needs this format, for instance. 

	Note: you can also convert n-grams or POS tags to a NB-friendly format with this.
	"""
	def __init__(self):
		self.count = 0
		self.id2word = {}
		self.word2id = {}
	def doc2bow(self, doc, full=False):
		counts = {}
		for token in doc:
			if token in self.word2id:
				id = self.word2id[token]
			else:
				id = self.count
				self.word2id[token] = id
				self.id2word[id] = token
				self.count += 1

			# the "full" flag just decides whether we return {id: count,...} or {token: count, ...}
			# both work for NB, but for some reason SVM expects a string to int mapping
			if full:
				counts[token] = counts.get(id, 0) + 1
			else:
				counts[id] = counts.get(id, 0) + 1
		return counts


class NaiveBayesTweetClassifier(TweetClassifier):
	"""
	A simple Naive Bayes classifier. Documents are tokenized and stemmed, and then converted to bag-of-words format.
	The preprocessed documents are then handled by NLTKs Naive Bayes classifier.
	"""
	def __init__(self, trainfile=None, datafile=None, outfile=None):
		super(NaiveBayesTweetClassifier, self).__init__(trainfile, datafile, outfile)
		self.dictionary = SimpleDict()
		self.scores = {}
		self.stemmer = PorterStemmer()

	def getFeatures(self, tweet):
		return self.getFeatures2(tweet)
		"""
		Replace this method to select different features than just bag-of-words representation of the whole tweet.
		This is probably the one piece of code we should work on most, since features basically decide whether we have a good or bad classifier.
		"""
		#tokens = string.lower(tweet.tweet.translate(string.maketrans("",""), string.punctuation)).split(" ")
		#tokens = [self.stemmer.stem(token) for token in tokens]
		#tokens = [token for token in tokens if not token[0:4] == "http"] #remove links
		#for stop in STOPWORDS:
	#		if stop in tokens:
#				tokens.remove(stop)
#		return self.dictionary.doc2bow(tokens)

	def getFeatures2(self, tweet):
		"""
		This is one attempt at getting different features (POS tagged bag-of-words), works rather poorly though.
		"""
		text = nltk.word_tokenize(tweet.tweet)
		return self.dictionary.doc2bow(nltk.pos_tag(text))

	def train(self,  trainfile=None):
		self.readTrainingData((trainfile or self.trainfile))
		print "getting features.."
		# the classifier expects a list of (feature_set, label) elements, where each feature_set is a dictionary of {feature_name: value, ...} mappings
		train_set = [(self.getFeatures(tweet), tweet.sentiment) for tweet in self.trainingTweets]
		print "training NB classifier"
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def classifyTweets(self, datafile=None, outfile=None):
		print "reading dataset"
		self.readDataset(datafile)

		print "classifying Tweets with naive bayes classifier"
		for tweet in self.evalTweets:
			# we use prob_classify so we can remember the scores. This means we could later on judge the certainty of a measurement, and if it's low, change the sentiment.
			res = self.classifier.prob_classify(self.getFeatures(tweet))
			self.scores[(tweet.id1,tweet.id2)] = res
			tweet.sentiment = res.max()

		print "writing results."
		self.writeResults(outfile)

class SVMTweetClassifier(TweetClassifier):
	"""
	A simple Naive Bayes classifier. Documents are tokenized and stemmed, and then converted to bag-of-words format.
	The preprocessed documents are then handled by NLTKs Naive Bayes classifier.
	"""
	def __init__(self, trainfile=None, datafile=None, outfile=None):
		super(SVMTweetClassifier, self).__init__(trainfile, datafile, outfile)
		self.dictionary = SimpleDict()
		self.scores = {}
		self.stemmer = PorterStemmer()

	def getFeatures(self, tweet):
		"""
		Replace this method to select different features than just bag-of-words representation of the whole tweet.
		This is probably the one piece of code we should work on most, since features basically decide whether we have a good or bad classifier.
		"""
		return self.getFeatures2(tweet)
		#tokens = string.lower(tweet.tweet.translate(string.maketrans("",""), string.punctuation)).split(" ")
		#tokens = [self.stemmer.stem(token) for token in tokens]
		#tokens = [token for token in tokens if not token[0:4] == "http"] #remove links
		#for stop in STOPWORDS:
	#		if stop in tokens:
#				tokens.remove(stop)
		#return self.dictionary.doc2bow(tokens, True)

	def getFeatures2(self, tweet):
		"""
		POS tag and take only nouns, verbs and adjectives
		"""
		text = nltk.word_tokenize(tweet.tweet)
		return self.dictionary.doc2bow([pos for pos in nltk.pos_tag(text) if pos[1] in ["NN","JJ","JJR","JJS","VBD","VBG","VBN" ,"VBP","VBZ" ,"RB"] ])

	def train(self,  trainfile=None):
		self.readTrainingData((trainfile or self.trainfile))
		print "getting features.."
		# the classifier expects a list of (feature_set, label) elements, where each feature_set is a dictionary of {feature_name: value, ...} mappings
		train_set = [(self.getFeatures(tweet), tweet.sentiment) for tweet in self.trainingTweets]
		print train_set
		print "training SVM classifier"
		self.classifier = SklearnClassifier(SVC(), sparse=True).train(train_set)

	def classifyTweets(self, datafile=None, outfile=None):
		print "reading dataset"
		self.readDataset(datafile)

		print "classifying Tweets with SVM classifier"
		
		# we use prob_classify so we can remember the scores. This means we could later on judge the certainty of a measurement, and if it's low, change the sentiment.
		res = self.classifier.batch_classify([self.getFeatures(tweet) for tweet in self.evalTweets])
		print "assigning sentiments"
		for idx, tweet in enumerate(self.evalTweets):
			tweet.sentiment = res[idx]

		#self.scores[(tweet.id1,tweet.id2)] = res
		#tweet.sentiment = res.max()

		self.writeResults(outfile)


class JointClassifier(TweetClassifier):
	"""
	This approach combines the NaiveBayesClassifier and WeightedTweetClassifier.
	NB is used mainly, but:
		neutral tweets can be reassigned to positive, if the WeightedTweetClassifier confidence is high
		negative tweets are reassigned positive, since the WeightedTweetClassifier is better at predicting those

	There isn't really a good reason for this other than that it slightly improves the f score.

	Also: this runs the other two classifiers at once, so we can easily compare their output.
	"""
	def __init__(self,dictfile=None, trainfile=None, datafile=None, outfile=None):
		self.wc = WeightedTweetClassifier(dictfile, trainfile, datafile, "outWeight.pred")
		self.nbc = NaiveBayesTweetClassifier(trainfile, datafile, "outNB.pred")
		#self.svmc = SVMTweetClassifier(trainfile, datafile, "outSVM.pred")
		self.outfile = outfile
	
	def train(self):
		self.wc.train()
		self.nbc.train()
		#self.svmc.train()
	
	def classifyTweets(self, outfile = None):
		self.wc.classifyTweets()
		self.nbc.classifyTweets()
		#self.svmc.classifyTweets()
		
		self.evalTweets = self.nbc.evalTweets

		for tweet in self.evalTweets:
			# find out what the weighted classifier thinks
			wcscore = self.wc.scores[(tweet.id1,tweet.id2)]
			wcsentiment = ("neutral" if abs(wcscore) < 0.01 else ( "negative" if wcscore < 0 else "positive"))

			# if the weighted classifier has a high certainty on a positive judgement, change neutral tweets to positive
			if wcscore > 0.1 and tweet.sentiment == "neutral":
				tweet.sentiment = "positive"

			# if NB says negative, but there were clearly positive terms present, change to positive (this was a pretty good improvement)
			if tweet.sentiment == "negative" and wcsentiment == "positive":
				tweet.sentiment = "positive"


		self.writeResults(outfile)



dictfile = "polar-dict/clues.tff.patched"
trainfile = "train_set.pred"
datafile = "dev_test_set.pred"


