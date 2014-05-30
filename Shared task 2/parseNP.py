import nltk
from nltk.corpus import conll2000

from nltk import Tree

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


tagtocat = {
            'NN' : 'NN',
            'NNP' : 'NN',
            'JJ' : 'CON',
            'NNS' : 'NN',
            'VB' : 'VB',
            'VBN' : 'VB',
            'VBG' : 'VB',
            'CD' : 'CD',
            'VBD' : 'VB',
            'RB' : 'Adj',
            'VBZ' : 'VB',
            'VBP' : 'VB',
            'IN' : 'CON',
            'NNPS' : 'NN',
            'JJR' : 'Adj',
            'JJS' : 'Adj',
            'DT' : 'Det',
            'RBR' : 'Adj',
            'CC' : 'CON',
            'WRB' : 'Adj',
            'FW' : 'NN'
        }

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

            current_sentence.append((token, tagtocat.get(pos, pos), bracket))

        sentences = [s for s in sentences if s]

        merged_sentences = [merge_to_tree(sentence) for sentence in sentences]
        merged_sentences = [m for m in merged_sentences if m]
        print "retrieved %s sentences, failed to read %s, managed to parse %s trees." % (len(sentences), failed, len(merged_sentences))
        return merged_sentences



def merge_to_tree(sent):
    result = "Tree('S',["
    for line in sent:
        openb = line[2].translate(None,"1234567890)_|")
        closeb = line[2].translate(None,"1234567890(_|")
        for x in range(len(openb)):
            result += "Tree('NP',[" 
        result += ("," if (not len(openb) and result[-1:] not in "([") else "") \
            + "('" + line[0].replace("'","\\'") + "','" + line[1].replace("'","\\'") +  "')"
        for x in range(len(closeb)):
            result += "])"
    result += "])"
    try:
        return eval(result.replace(")T","),T"))
    except Exception, e:
        print "failed: %s : %s -> %s" % (e, sent, result)
        return None


def rebuild_sentence(sentence):  #fuckyeah
    return "".join([(" " + token if token[0] not in list(".,?'\"!") else token) for token, pos, _ in sentence])[1:]


idx = 0
def traverse(t, depth):
    global idx
    id = 0#random.randint(1,500)
    if not hasattr(t, "node"):
        idx += 1
        ending = depth * ("("+str(id)) if depth else "_"
        return str(idx)+"    "+t[0]+"    "+t[1]+("    %s\n" % ending)
    else:
        depth += 1
        # Now we know that t.node is defined
        start = ""
        for i, child in enumerate(t[:-1]):
            if i == 0:
                start += traverse(child, depth)
            else:
                start += traverse(child, 0)
        ending = str(id)+")"
        start += traverse(t[-1], 0).strip("_\n") +ending+"\n"
        return start
        

def write_results(results, outfile):
    with open(outfile) as f:
        for sentence in results:
            res = traverse(sentence, 0)
            f.write(res + "\n")

def get_tag(chunk):
    return chunk.node if hasattr(chunk, "node") else chunk[1]


def recursive_np_chunk_features(treelist, i, history):
    tag = treelist[i]
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
            "prevtag+tag": "%s+%s" % (prevtag, tag),  
            "tag+nexttag": "%s+%s" % (tag, nexttag),
            "tags-since-dt": tags_since_dt(sentence, i)}

class MLRecursiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_chunks):
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

class MLRecursiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_chunks):
        tagged_sents = [[((word, tag), chunk) for (word, tag, chunk) in tree2iob_tagged_subtrees(sent)] for sent in train_sents]
        self.chunker_tagger = MLRecursiveNPChunker(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return iobplus2tree(conlltags)

def tree2chunklist(tree):
    result = []
    if hasattr(tree, "node"):
        result.append((tree.node, [get_tag(c) for c in tree]))
        for child in tree:
            result += tree2chunklist(child)
    else:
        pass
        #result.append((tree[1], tree[0]))
    return result

def tree2iob_tagged_subtrees(tree):
    result = []
    #print "now on: \n", tree
    if not isinstance(tree, Tree):
        return result

    try:
        result.append(nltk.chunk.util.tree2conlltags(tree))
    except ValueError, e:
        #print
        #print e
        #print tree
        for child in tree:
            result += tree2iob_tagged_subtrees(child)
        newTree = Tree(tree.node, [(("<"+get_tag(c)+">", get_tag(c)) if isinstance(c, Tree) else (c[0], get_tag(c))) for c in tree])
        #print newTree
        result.append(nltk.chunk.util.tree2conlltags(newTree))
    return result

class GrammarRecursiveNPChunker(nltk.TaggerI):
    def __init__(self, train_sents):
        self.chunker = ConsecutiveNPChunker(train_sents)
        

    def parse(self, sentence):
        grammar = generate_grammar(sentence)
        self.recursive_np_chunker = nltk.RecursiveDescentParser(grammar)

        res = self.chunker.parse(sentence)
        print "initial: \n%s\n" % res.pprint()

        current = res
        last = None
        while current != last:
            last = current
            current = self.recursive_np_chunker.parse(last)
            print "intermediate: \n%s\n" % current.pprint()

        print "final: \n%s\n" % current.pprint()
        return current
        
def parse(sent):
    return rc.chunker.parse(ie_preprocess(sent)[0])

def parse_rec(sent):
    return rc.parse(ie_preprocess(sent)[0])

def traverse_tags(t):
    try:
        t.node
    except AttributeError:
        return [t]
    else:
        # Now we know that t.node is defined
        tags  = [t.node]
        for child in t:
            tags += traverse_tags(child)
        return tags

if __name__ == "__main__":
    rc = RecursiveNPChunker(train_sents)

