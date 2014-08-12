from pattern.de import parsetree
import io
import os
import nltk
import freebase

ffailed = 0
entities = dict()
classifier = None

from nltk.stem.snowball import GermanStemmer

def parsetree_w(string):
    t = parsetree(string)[0]

    deleted = 0

    for idx, word in enumerate(t.words[:]):
        if word.string == "-":
            if idx == 0:
                t.words[idx+1].string = "-" + t.words[idx+1].string
                del t.words[idx - deleted]
                deleted += 1
            else:
                t.words[idx-1].string += "-"
                del t.words[idx - deleted]
                deleted += 1
        #if word.string == "'":
        #    if idx == 0:
        #        t.words[idx+1].string += "'"
        #        del t.words[idx]
        #    else:
        #        t.words[idx-1].string = "'" + t.words[idx+1].string
        #        del t.words[idx]


    return t

vornamen = set()
with open("vornamen.txt", "r") as vornamenf:
    for name in vornamenf.readlines():
        vornamen.add(name.strip())

nachnamen = set()
with open("nachnamen.txt", "r") as nachnamenf:
    for name in nachnamenf.readlines():
        nachnamen.add(name.strip())

for filename in os.listdir("lists"):
    with open("lists/"+filename, "r") as f:
        for line in f.readlines():
            tag = line[:line.find(" ")]
            if tag == "MISC":
                tag = "OTH"
            if tag == "MISC":
                tag = "OTH"
            entity = line[line.find(" ") + 1:-1]
            entities[entity] = tag


def _check_NE_yeah(gram):
    tag = entities.get(" ".join(gram), "O")

    if tag == "O":
        if len(gram) == 2:
            first, last = gram
            if first in vornamen and last in nachnamen:
                tag = "PER"

    if tag == "O":
        try:
            tag = entities.get(" ".join([GermanStemmer().stem(g) for g in gram]), "O")
        except:
            tag = entities.get(" ".join([GermanStemmer().stem(g.decode(encoding="UTF-8")) for g in gram]), "O")

    return tag


def check_NE(candidate, chunk):
    found = []

    chunkstrs = [convert(word.string) for word in chunk.words]

    if True:  # tag == "O":
        for n in reversed(range(1, len(chunk) + 1)):
            for gram in ngrams(chunkstrs, n):
                tag = _check_NE_yeah(gram)
                if tag != "O":
                    found.append(gram)
                    break
            if tag != "O":
                break

    if found:
        candidate_position = chunkstrs.index(candidate)
        entity_start_position = chunkstrs.index(found[0][0])
        entity_stop_position = chunkstrs.index(found[0][-1])
        if candidate_position == entity_start_position:
            tag = "B-" + tag
        elif entity_start_position < candidate_position <= entity_stop_position:
            tag = "I-" + tag
        else:
            tag = "O"

    return tag

def ngrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


class Sentence:
    def __init__(self, title, text):
        self.title = title
        self.text = text
        self.tags = []


def read_file(filename, train=False):
    with io.open(filename, "r", encoding='utf8') as f:
        sentences = list()
        current_sentence = list()
        failed = 0
        #print "opened"

        for idx, line in enumerate(f.readlines()):
            if line[0] == "#":
                if train:
                    continue
                else:
                    sentences.append(line)
                    continue
            if line.strip() == "":
                if len(current_sentence):
                    sentences.append(current_sentence)
                current_sentence = []
                continue

            idx, token, iob1, iob2 = line.split()

            current_sentence.append((token, iob1, iob2))

        sentences = [s for s in sentences if s]
        s = None

        res = []
        for sentence in sentences:
            if type(sentence) == list:
                try:
                    s.text = " ".join([w for w, i1, i2 in sentence])
                    s.tags = [(i1, i2) for w, i1, i2 in sentence]
                except:
                    print sentence, type(sentence)
                    raw_input()

            else:
                if s:
                    res.append(s)
                s = Sentence(sentence, "")

        print "returning sentence list of length %s" % len(res)
        #print "couldn't read %s sents" % failed
        return res


def convert(stuff):
    return stuff.encode("UTF-8")


def handleSentence(sentence):
    global ffailed

    result = convert(sentence.title) + "\n"

    last_chunk = None

    idx = 1
    tree = parsetree_w(sentence.text)

    for ii, word in enumerate(tree):
        wordstr = convert(word.string)

        if word.chunk is None:
            TAG = "O"
        elif word.chunk.type == "NP":
            if word.chunk == last_chunk:
                TAG = check_NE(wordstr, word.chunk)
            else:
                TAG = check_NE(wordstr, word.chunk)
                last_chunk = word.chunk

        else:
            TAG = "O"
        try:
            result += "%s\t%s\t%s\t%s\t%s\tO\n" % (idx, wordstr,
                                                       convert(sentence.tags[idx-1][0]),
                                                       convert(sentence.tags[idx-1][1]),
                                                       TAG)
        except:
            ffailed += 1

        idx += 1

    return result


class NERChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)

            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = ner_features(untagged_sent, i, history, )
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = ner_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


def ner_features(sentence, i, history):
    # TODO: try using TreeTagger's POS tag
    wordO = sentence[i]
    word = wordO.string
    pos = wordO.pos
    stemmed = GermanStemmer().stem(word)

    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        last = "<START>"
        prevstemmed = "<START>"
    else:
        last = history[-1]
        prevword = sentence[i-1].string
        prevpos = sentence[i-1].pos
        prevstemmed = GermanStemmer().stem(sentence[i-1].string)

    chunk = []
    if not wordO.chunk:
        chunk.append("START")
        knowledge_sources = "O"
    else:
        knowledge_sources = check_NE(convert(wordO.string), wordO.chunk)
        chunk = [w.string for w in wordO.chunk]


    stem_is_word = stemmed == word.lower()

    knowledge_sources_stemmed = _check_NE_yeah([stemmed])

    return {"knowledge": knowledge_sources,
            "knowledge_lemma": knowledge_sources_stemmed,
            "history": "+".join(history)[-2:],
            "pos": pos,
            "word": word,
            "stemmed": stemmed}
            #"chunk": "+".join(chunk),
            #"prevpos": prevpos}
            #"prevstemmed": prevstemmed
            #"nextpos": nextpos,
            #"prevpos+pos": "%s+%s" % (prevpos, pos)}
            #"pos+nextpos": "%s+%s" % (pos, nextpos)}


def run_experiment(ML=False):
    global classifier
    global ffailed
    ffailed = 0
    with io.open("outfile_spelchek.tsv", "w", encoding='utf8') as f:
        if ML:
            print "using ML approach"
            train_sents = []
            print "reading train data, building featuresets.."
            sents = read_file("NER-de-train.tsv")
            for i, sent in enumerate(sents):
                if i % 500 == 0:
                    print i
                tags = [t[0] for t in sent.tags]
                tree = parsetree_w(sent.text)
                train_sent = zip(tree.words, tags)
                train_sents.append(train_sent)
            print "training model.."
            classifier = NERChunkTagger(train_sents)

            print "running evaluation"
            evalsents = read_file("NER-de-dev.tsv")
            for sent in evalsents:
                # classify sentence
                pos_tagged_words = parsetree_w(sent.text).words
                classified_sent = classifier.tag(pos_tagged_words)
                result = convert(sent.title) + "\n"
                for idx, (word, TAG) in enumerate(classified_sent):
                    try:
                        assert(len(sent.tags) == len(classified_sent))
                    except AssertionError:
                        #print sent.tags
                        #print classified_sent
                        #print pos_tagged_words
                        pass
                    wordstr = convert(word.string)
                    try:
                        result += "%s\t%s\t%s\t%s\t%s\tO\n" % (idx + 1, wordstr,
                                                           convert(sent.tags[idx][0]),
                                                           convert(sent.tags[idx][1]),
                                                           convert(TAG))
                    except:
                        ffailed += 1
                        pass
                try:
                    f.write(unicode(result, "UTF-8") + "\n")
                except TypeError:
                    f.write(result + "\n")
            print "%s failed" % (ffailed)

        else:
            print "using knowledge based approach"
            sents = read_file("NER-de-dev.tsv")
            for sent in sents:
                result = handleSentence(sent)
                f.write(unicode(result, "UTF-8") + "\n")
            print "%s failed" % ffailed
