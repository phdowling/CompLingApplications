__author__ = 'dowling'

import logging
import gensim
import bz2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from bs4 import BeautifulSoup
from collections import defaultdict
from stemming.porter2 import stem
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from numpy import array
from nltk import NaiveBayesClassifier
import random
from itertools import groupby

STOPWORDS = stopwords.words("english")

WORD_IDS_FILE = "wikipedia/wiki_en_wordids.txt"
#WORD_IDS_FILE = "holist.dict"
#MM_CORPUS_FILE = 'wiki_en_tfidf.mm'
#MM_CORPUS_FILE = 'holist.mm'
MODEL_DIRECTORY = "wikipedia/"
MODEL_NAME = "wikipedia_300.lsa"
TFIDF_FILE = "wiki_en_tfidf_model.tfidf"
WN_MAPPING_FILE = "train/EnglishLS.dictionary.mapping.xml"

TRAIN_FILE = "train/EnglishLS.train"
TEST_FILE = "test/EnglishLS.test"

OUTFILE = "outfiles/SPELCHEK.model_%s.topics_%s.cir_%s.multsenseinclude_%s.out"

RANGE_TOP = 50

USE_CLASSIFIERS = False

INCLUDE_RANGE = -1


def most_frequent(l, top=10, include_range=0):
    """
    return the most frequently occurrind sense in the top n similar documents
    """
    counter = defaultdict(int)
    for sense, similarity in l[:top]:
        counter[sense] += 1
    items = counter.items()
    items.sort(key=lambda k: -k[1])
    most_freq = items[0][1]
    result = [items[0][0]]
    if include_range != -1:
        for item in items[1:]:
            if most_freq - item[1] <= include_range:
                result.append(item[0])

    return result


def best_average(similarities, top=10):
    averages = {}
    similarities = sorted(similarities, key=lambda (k, v): -v)[:top]
    assert len(similarities) <= top
    groups = groupby(similarities[:], key=lambda (k, v): k)
    for senseid, similarityIterator in groups:
        asList = list(similarityIterator)
        #print asList
        sims = [sim for sid, sim in asList]
        averages[senseid] = sum(sims) / float(len(sims))
    best = sorted(averages.items(), key=lambda (k, v): -v)
    best = best[0][0]
    return [best]


def best_weighted_average(similarities, top=10):
    averages = {}
    similarities = sorted(similarities, key=lambda (k, v): -v)[:top]
    assert len(similarities) <= top
    groups = groupby(similarities[:], key=lambda (k, v): k)
    for senseid, similarityIterator in groups:
        groupAsList = list(similarityIterator)
        #print asList
        sims = [sim for sid, sim in groupAsList]
        averages[senseid] = len(groupAsList) * sum(sims) / top * float(len(sims))
    best = sorted(averages.items(), key=lambda (k, v): -v)
    best = best[0][0]
    return [best]


class LSAWSD(object):

    def __init__(self):
        self.model = None
        self.id2word = None
        self.equivalentSenses = None
        self.senses = dict()
        self.classifiers = {}
        self.saved_similarities = None
        self.model_type = None
        self.load_similarities = False
        self.use_tfidf = True
        self.tfidf = None
        self.use_whole_context = False
        self.local_radius = 10

    def convert(self, vector):
        """
        convert a sparse vector into a numpy array
        """
        res = []
        for x in range(self.model.num_topics):
            res.append(0)

        for dim, val in vector:
            res[dim] = val

        return numpy.array(res)

    def train(self):
        if self.use_tfidf:
            print "reading TF-IDF model"
            self.tfidf = gensim.models.TfidfModel.load(MODEL_DIRECTORY + TFIDF_FILE)
        print "training topic model."
        self.train_LSA()
        print "reading sense data"
        self.read_sense_data()

        if USE_CLASSIFIERS:
            print "training classifiers"
            self.train_classifiers()

    def train_LSA(self):
        self.id2word = gensim.corpora.Dictionary.load_from_text(WORD_IDS_FILE)

        modelname = MODEL_DIRECTORY + MODEL_NAME
        print "loading model from %s" % modelname
        if modelname.endswith(".lsa"):
            self.model = gensim.models.LsiModel.load(modelname)
            self.model_type = "LSA"
        elif modelname.endswith(".lda"):
            self.model = gensim.models.LdaModel.load(modelname)
            self.model_type = "LDA"
        else:
            raise ValueError("Unknown model type: %s" % modelname)

    def read_sense_data(self):
        try:
            if self.use_tfidf:
                with open("save_sense_data_tfidf_%s" % MODEL_NAME, "r") as s:
                    self.senses = eval(s.read())
            else:
                with open("save_sense_data%s" % MODEL_NAME, "r") as s:
                    self.senses = eval(s.read())
            print "read training senses from file"
        except IOError:
            with open(TRAIN_FILE) as f:
                soup = BeautifulSoup(f.read())
                for lexelt in soup.findAll("lexelt"):
                    word = lexelt["item"]
                    self.senses[word] = {}
                    for instance in lexelt.findAll("instance"):
                        text = instance.context.text
                        for answer in instance.findAll("answer"):
                            senseId = answer["senseid"]
                            if senseId not in self.senses[word]:
                                self.senses[word][senseId] = []
                            vectorized = self.convert(self.model[self.preprocess(text)])
                            self.senses[word][senseId].append(vectorized)
            if self.use_tfidf:
                with open("save_sense_data_tfidf_%s" % MODEL_NAME, "w") as f:
                    f.write(str(self.senses))
            else:
                with open("save_sense_data%s" % MODEL_NAME, "w") as f:
                    f.write(str(self.senses))
            print "newly evaluated training senses"

    def train_classifiers(self):
        for word in self.senses:
            train_set = []
            for senseId in self.senses[word]:
                for lsa_vector in self.senses[word][senseId]:
                    train_set.append([dict(lsa_vector), senseId])
            self.classifiers[word] = NaiveBayesClassifier.train(train_set)

    def preprocess(self, text):
        tokens = text.split()
        #tokens = [stem(token) for token in tokens if token not in STOPWORDS]
        if self.use_tfidf:
            return self.tfidf[self.id2word.doc2bow(tokens)]
        else:
            return self.id2word.doc2bow(tokens)

    def compute_similarities(self, word, context):
        prep = self.preprocess(context)
        vector = self.convert(self.model[prep])
        similarities = []

        for senseId in self.senses[word]:
            for contextVector in self.senses[word][senseId]:
                similarity = float(cosine_similarity(vector, contextVector))
                similarities.append((senseId, similarity))

        similarities.sort(key=lambda x: -x[1])

        return similarities

    def load_test_instances(self):
        loaded_instances = []
        with open(TEST_FILE) as f:
            soup = BeautifulSoup(f.read())
            lexelts = soup.findAll("lexelt")
            for lexelt in lexelts:
                word = lexelt["item"]
                instances = lexelt.findAll("instance")
                for instance in instances:
                    instance_id = instance["id"]
                    if self.use_whole_context:
                        context = instance.context.text
                    else:
                        readableWord = stem(word[:word.find(".")])
                        context = instance.context.text.split(" ")
                        try:
                            found = False
                            for i, w in enumerate(context):
                                if stem(w) == readableWord:
                                    idx = i
                                    found = True
                                    break
                            assert found
                            #idx = context.index(readableWord)
                            fromidx = idx - self.local_radius
                            fromidx = fromidx if fromidx >= 0 else 0
                            toidx = idx + self.local_radius
                            toidx = toidx if toidx < len(context) else -1
                            context = " ".join(context[fromidx: toidx])
                        except AssertionError:

                            print "failed radius evaluation on %s" % instance_id
                            print readableWord
                            print context
                            context = instance.context.text

                    loaded_instances.append((instance_id, word, context))

        return loaded_instances

    def run_evaluation(self, loaded_instances):
        save = []
        frequency_results = defaultdict(list)

        for i, (instance_id, word, context) in enumerate(loaded_instances):
            if i % 250 == 0:
                print "Evaluated %s instances.." % i

            if self.load_similarities:
                similarities = self.saved_similarities[i]
            else:
                similarities = self.compute_similarities(word, context)

            # similarities holds a sorted list ranking the different possible senses FOR EACH DOCUMENT of that sense
            # [(senseID, similarity)...]

            if not self.load_similarities:
                save.append(similarities)
            for top in range(1, RANGE_TOP):
                word_senses = most_frequent(similarities, top=top, include_range=INCLUDE_RANGE)
                #word_senses = best_average(similarities, top=top)
                #word_senses = best_weighted_average(similarities, top=top)
                frequency_results[top].append("%s %s %s" % (word, instance_id, " ".join(word_senses)))
        print "Evaluation finished. Now writing outfiles."
        savestring = ("_tfidf" if self.use_tfidf else "") + \
                     ("_radius%s" % self.local_radius if not self.use_whole_context else "")
        if not self.load_similarities:
            with open("save_sense_similarities%s_%s" % (savestring, MODEL_NAME), "w") as sf:
                sf.write(str(save))

        for top in range(1, RANGE_TOP):

            fname = OUTFILE % (self.model.num_topics, savestring + self.model_type, "top" + str(top), INCLUDE_RANGE)
            with open(fname, "w") as outfile:
                print "writing %s.." % fname
                outfile.write("\n".join(frequency_results[top]) + "\n")

    def run_experiment(self):
        try:
            savestring = ("_tfidf" if self.use_tfidf else "") + \
                         ("_radius%s" % self.local_radius if not self.use_whole_context else "")
            with open("save_sense_similarities%s_%s" % (savestring, MODEL_NAME), "r") as s:
                self.saved_similarities = eval(s.read())
            self.load_similarities = True
        except IOError:
            self.load_similarities = False

        loaded_instances = self.load_test_instances()

        print "got %s total instances, evaluating." % len(loaded_instances)

        self.run_evaluation(loaded_instances)