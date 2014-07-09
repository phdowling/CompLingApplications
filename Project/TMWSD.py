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

STOPWORDS = stopwords.words("english")

#WORD_IDS_FILE = "wiki_en_wordids.txt"
WORD_IDS_FILE = "holist.dict"
#MM_CORPUS_FILE = 'wiki_en_tfidf.mm'
MM_CORPUS_FILE = 'holist.mm'

MODEL_NAME = "holist.lsa"
WN_MAPPING_FILE = "train/EnglishLS.dictionary.mapping.xml"

TRAIN_FILE = "train/EnglishLS.train"
TEST_FILE = "train/EnglishLS.test"

NUM_TOPICS = 300

def convert(vector):
    res = []
    for x in range(NUM_TOPICS):
        res.append(0)

    for dim, val in vector:
        res[dim] = val

    return res


class LSAWSD(object):

    def __init__(self):
        self.model = None
        self.id2word = None
        self.equivalentSenses = None
        self.senses = defaultdict(lambda: defaultdict(list))

    def train(self):
        print "training LSA model."
        self.train_LSA()
        print "reading sense data"
        self.read_sense_data()

    def train_LSA(self):
        #self.id2word = gensim.corpora.Dictionary.load_from_text(WORD_IDS_FILE)
        self.id2word = gensim.corpora.Dictionary.load(WORD_IDS_FILE)

        mm = gensim.corpora.MmCorpus(MM_CORPUS_FILE)

        try:
            self.model = gensim.models.LsiModel.load(MODEL_NAME)
        except IOError:
            self.model = gensim.models.LsiModel(corpus=mm, num_topics=NUM_TOPICS, id2word=self.id2word)
            self.model.save(MODEL_NAME)
            # lda = gensim.models.LdaModel(corpus=mm, id2word=id2word, num_topics=300, chunksize=10000)

    def read_sense_data(self):
        #self.read_wordnet_mappings()
        with open(TRAIN_FILE) as f:
            soup = BeautifulSoup(f.read())
            for lexelt in soup.findAll("lexelt"):
                word = lexelt["item"]
                for instance in lexelt.findAll("instance"):
                    text = instance.context.text
                    for answer in instance.findAll("answer"):
                        #senseId = self.equivalentSenses.get(answer["senseid"], answer["senseid"])
                        senseId = answer["senseid"]
                        lsa = self.model[self.preprocess(text)]
                        self.senses[word.split(".")[0]][senseId].append(lsa)
        #print self.senses

    #def read_wordnet_mappings(self):
    #    self.equivalentSenses = dict()
    #    with open(WN_MAPPING_FILE) as f:
    #        soup = BeautifulSoup(f.read())
    #        for lexelt in soup.findAll("lexelt"):
    #            word = lexelt["item"]
    #            for sense in lexelt.findAll("sense"):
    #                if sense["source"] != "wn":
    #                    try:
    #                        id = sense["id"]
    #                        wn = sense["wn"]
    #                        self.equivalentSenses[wn] = id
    #                    except:
    #                        print sense

    def preprocess(self, text):
        tokens = text.split()
        tokens = [stem(token) for token in tokens if token not in STOPWORDS]
        return self.id2word.doc2bow(tokens)

    def evaluate(self, word, context):
        prep = self.preprocess(context)
        print prep
        vector = self.model[prep]
        best = None
        maximum = 0
        for senseId in self.senses[word]:
            for contextVector in self.senses[word][senseId]:
                similarity = cosine_similarity(convert(vector), convert(contextVector))
                if similarity > maximum:
                    maximum = similarity
                    best = senseId
        return best

    def run_experiment(self):
        with open(TEST_FILE) as f:
            soup = BeautifulSoup(f.read())
            for lexelt in soup.findAll("lexelt"):
                word = lexelt["item"]
                for instance in lexelt.findAll("instance"):
                    text = instance.context.text
                    for answer in instance.findAll("answer"):
                        senseId = answer["senseid"]
                        self.senses[word][senseId].append(self.model[self.preprocess(text)])