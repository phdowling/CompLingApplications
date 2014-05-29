import nltk
from nltk.corpus import conll2000

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

grammar = """
   NP: {N}          
   PP: {<IN><NP>}               
   VP: {<VB.*><NP|PP|CLAUSE>+$} 
   CLAUSE: {<NP><VP>}           
   """

grammar = '''
 NP: {<DT>? <JJ>* <NN>*} # NP
 P: {<IN>}           # Preposition
 V: {<V.*>}          # Verb
 PP: {<P> <NP>}      # PP -> P NP
 VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
 '''
nltk.RegexpParser(grammar)

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
            if bracket[0] == "(":
                pass
            if bracket[:-1] == ")":
                pass
            remember = bracket
            chunk = []


def rebuild_sentence(sentence):  #fuckyeah
    return "".join([(" " + token if token[0] not in list(".,?'\"!") else token) for token, pos, _ in sentence])[1:]


def write_results(results, outfile):
    pass


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

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
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
        tagged_sents = [[((word, tag), chunk) for (word, tag, chunk) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        print "conlltags are: %s" % conlltags
        return nltk.chunk.util.conlltags2tree(conlltags)

def get_tag(chunk):
    return chunk.node if hasattr(chunk, "node") else chunk[1]

def recursive_np_chunk_features(treelist, i, history):
    # treelist: 
    #[
    # ('NP', Tree('NP', [('The', 'DT'), ('quick', 'NN'), ('brown', 'NN'), ('fox', 'NN')])),
    # ('IN', ('from', 'IN')), 
    # ('NP', Tree('NP', [('mars', 'NNS')])), 
    # ('VBD', ('jumped', 'VBD')), 
    # ('IN', ('over', 'IN')),
    # ('NP', Tree('NP', [('the', 'DT'), ('fence', 'NN')])), 
    # ('.', ('.', '.'))
    #]
    tag, content = treelist[i]
    if i == 0:
        prevtag, prevcontent = "<START>", "<START>"
    else:
        prevtag, prevcontent = treelist[i-1]
    if i == len(treelist)-1:
        nexttag, nextcontent = "<END>", "<END>"
    else:
        nexttag, nextcontent = treelist[i+1]
    return {"tag": tag,
            "prevtag": prevtag,
            "nexttag": nexttag,
            "prevpos+pos": "%s+%s" % (prevpos, pos),  
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}



class RecursiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_chunks):
        train_set = []
        for tagged_chunk in train_chunks:
            history = []
            for i, (chunk, tree) in enumerate(tagged_chunk):
                featureset = recursive_np_chunk_features(tree, i, history)


c = ConsecutiveNPChunker(train_sents)
def parse(sent):
    return c.parse(ie_preprocess(sent)[0])


