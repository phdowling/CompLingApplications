import nltk
from nltk.corpus import conll2000

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

grammar = nltk.parse_cfg("""
S -> NP VP

NP -> N | Det N | Det Adj N | N PP | Det N PP | Det Adj N PP
NP -> NP conj NP

PP -> P | P NP


VP -> V Adj | V NP | V S | V NP PP
VP -> VP conj VP
""")

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),  
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

def read_file(filename):  # this looks a bit messy but it works like a charm
    with open(filename) as f:
        sentences = list()
        current_sentence = list()
        skip = False
        failed = 0
        print "opened"

        for idx, line in enumerate(f.readlines()):
            if line[0] == "#":
                continue
            if line.strip() == "":
                skip = False
                sentences.append(current_sentence)
                current_sentence = []
                continue
            if skip: # ignore a sentence if any term couldn't be parsed
                print "skipping ahead line %s: %s" % (idx,line.split())
                continue

            try:
                idx, token, pos, bracket = line.split()
            except:
                print "failed to read line %s: %s" % (idx,line.split())
                print "discarding sentence %s" % (current_sentence)
                current_sentence = []
                skip = True
                failed += 1
                continue

            current_sentence.append((token, pos, bracket))

        sentences = [s for s in sentences if s]
        #merged_sentences = [merge_chunks(sentence) for sentence in sentences]
        print "retrieved %s sentences, failed to read %s." % (len(sentences), failed)
        return sentences

def merge_chunks(sentence):
    res = []
    chunk = []
    remember = None
    for term, pos, bracket in sentence:
        if bracket == "_":
            res.append((term, pos))
        else: 
            remember = bracket
            chunk = []


def rebuild_sentence(sentence):  #fuckyeah
    return "".join([(" "+ token if token[0] not in list(".,?'\"!") else token) for token, pos in sentence])[1:]

def write_results(results, outfile):
    pass

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        #self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='iis', trace=0)  # megam
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for idx, word in enumerate(sentence):
            featureset = npchunk_features(sentence, idx, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((word,tag),chunk) for (word,tag,chunk) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.util.conlltags2tree(conlltags)



c = ConsecutiveNPChunker(train_sents)
def parse(sent):
    return c.parse(ie_preprocess(sent)[0])
