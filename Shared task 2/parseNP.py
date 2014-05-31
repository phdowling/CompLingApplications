import nltk
from nltk.corpus import conll2000

from nltk import Tree

TRAIN_SENTS = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

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
            "tags-since-dt": tags_since_dt(treelist, i)}

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


class MLRecursiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_chunks):
        train_set = []
        for tagged_chunk in train_chunks:
            #print tagged_sent
            untagged_sent = [(w, pt) for (w,pt), t in tagged_chunk]
            revised = []
            for w, pt in untagged_sent:
                if isinstance(w, Tree):
                    revised.append((pt, pt))
                else:
                    revised.append((w,pt))
            untagged_sent = revised
            #print untagged_sent
            history = []
            for i, ((word, pos_tag), chunk_tag) in enumerate(tagged_chunk):
                featureset = recursive_np_chunk_features(untagged_sent, i, history)
                train_set.append((featureset, chunk_tag))
                history.append(chunk_tag)
        #self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='iis', trace=0)  # megam
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        placeholders = {}
        treenodes = set()
        newsentence = []
        #replace trees with (hashable) placeholders
        for i, thing in enumerate(sentence):
            if isinstance(thing, Tree):
                placeholders[i] = thing
                newsentence.append((("<%s>" % i), thing.node))
                treenodes.add(thing.node)
            else:
                newsentence.append(thing)
        sentence = newsentence
        print sentence

        # classify
        history = []
        for idx, word in enumerate(sentence):
            featureset = recursive_np_chunk_features(sentence, idx, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        res =  zip(sentence, history)

        # retrieve and replace the placeholders
        newres = []
        for (w,t),c in res:
            if w[0] == "<" and w[2] == ">":
                i = int(w[1])
                retrieve = placeholders[i]
                newres.append(((retrieve,t),c))
            else:
                newres.append(((w,t),c))

        return newres

class MLRecursiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_chunks):
        tagged_chunks = []
        count = 0
        for chunk in train_chunks:
            if count % 500 == 0:
                print "read %s chunks.." % count
            tagged_chunks.append([((word, tag), chunk) for (word, tag, chunk) in tree2conlltags(chunk)])
            count += 1
        print "starting training"
        self.chunk_tagger = MLRecursiveNPChunkTagger(tagged_chunks)

    def parse(self, sentence):
        tagged_sent = self.chunk_tagger.tag(sentence)
        print tagged_sent
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sent]
        return conlltags2tree(conlltags)


data = read_file("en_train_s.txt")
chunks = []
for d in data:
    chunks += get_tree_levels(d)

chunks = [c for c in chunks if len(c) > 2]
mlr = MLRecursiveNPChunker(chunks)
cnp = ConsecutiveNPChunker(TRAIN_SENTS)

def parse(sentence):
    last = None
    current = cnp.parse(sentence)
    #current = sentence
    print "seed: %s" % current
    iteration = 0
    while current != last:
        iteration += 1
        last = current
        current = mlr.parse(current)
        print "iter %s: %s" % (iteration,current)
        raw_input()
    return current

def reduce_nps(sentence):
    """
    take any occurrences of NP trees that contain only one  NP tree and reduce them
    """
    pass
def get_tree_levels(tree):
    res = []
    last = None
    current = tree
    level = 1
    while last != current:
        res.append(current)
        last = current
        current = get_tree_level(tree, level)
        level += 1
        
    return res

def get_tree_level(tree, level):
    return Tree("S", _get_tree_level(tree, level))

def _get_tree_level(tree, level):
    # TODO: group subtrees by level!
    result = []
    if isinstance(tree, Tree):
        if level == 0:
            result += [c for c in tree]
        else:
            for child in tree:
                result += _get_tree_level(child, level-1)
    else:
        result.append((tree[0], tree[1]))

    return result

def tree2conlltags(t):
    """
    Return a list of 3-tuples containing ``(word, tag, IOB-tag)``.
    Convert a tree to the CoNLL IOB tag format.

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: list(tuple)
    """

    tags = []
    for child in t:
        try:
            category = child.node
            prefix = "B-"
            for contents in child:
                if isinstance(contents, Tree):
                    tags.append((contents, contents.node, prefix+contents.node))
                else:
                    tags.append((contents[0], contents[1], prefix+category))
                prefix = "I-"
        except AttributeError:
            tags.append((child[0], child[1], "O"))
    return tags

def conlltags2tree(sentence, chunk_types=('NP','PP','VP'),
                   root_label='S', strict=False):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, postag, chunktag) in sentence:
        #print
        #print word, postag, chunktag
        #print 
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word,postag))
        elif chunktag.startswith('B-'):
            if isinstance(word, Tree):
                tree.append(Tree(chunktag[2:], [word]))
            else:
                tree.append(Tree(chunktag[2:], [(word,postag)]))
        elif chunktag.startswith('I-'):
            if (len(tree)==0 or not isinstance(tree[-1], Tree) or tree[-1].node != chunktag[2:]):
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word,postag)]))
            else:
                if isinstance(word, Tree):
                    tree[-1].append(word)
                else:
                    tree[-1].append((word,postag))
        elif chunktag == 'O':
            if isinstance(word, Tree):
                print "triggered"
                tree.append(word)
            else:
                tree.append((word,postag))
        else:
            raise ValueError("Bad conll tag %r" % chunktag)
    return tree


def tree2iobplus(tree):
    sentence = []
    _tree2iobplus(tree, '', sentence, True)
    return sentence

def _tree2iobplus(tree, iobtag, sentence, firstinlvl):
    if isinstance(tree, Tree):
        if tree.node == 'NP':
            iobtag += 'B'

        firstinlvl = True
        for child in tree:
            _tree2iobplus(child, iobtag, sentence, firstinlvl)
            if firstinlvl: 
                iobtag = 'I' * len(iobtag)
                firstinlvl = False
    elif isinstance(tree, tuple) and len(tree) == 2:
        name = tree[0]
        postag = tree[1]
       
        iobtag0 = None
        if len(iobtag) == 0:
            iobtag0 = 'O'
        else:
            iobtag0 = iobtag

        sentence.append((name, postag, iobtag0))
    else:
        raise Exception("illegal path")

# das sollte funktionieren, habs aber nur mit dem einem
# satz getestet. musst also noch mit einem anderen satz
# testen um 100 prozentig sicher zu sein :)
#
# sentence == np.tree2iobplus(np.iobplus2tree(sentence))
def iobplus2tree(sentence):
    tree = root = Tree('S', [])
    previob = ''
    S = [] 
    for _word in sentence:
        word, postag, iobtag = _word

        for _ in xrange(len(previob) - len(iobtag)):
            tree = S.pop()

        if iobtag == 'O':
            if any(S):
                tree = S.pop()
        else:
            for c in iobtag:
                if c == 'B':
                    subtree = Tree('NP', [])
                    tree.append(subtree)
                    S.append(tree)
                    tree = subtree

        tree.append((word, postag))
        previob = iobtag

    return root

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
    rc = MLRecursiveNPChunker(train_sents)


class ConsecutiveNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        #self.classifier = nltk.MaxentClassifier.train( train_set, algorithm='megam', trace=0)
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.util.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.util.conlltags2tree(conlltags)


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

