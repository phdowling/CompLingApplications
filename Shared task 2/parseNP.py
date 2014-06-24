import nltk
from nltk.corpus import conll2000

from nltk import Tree

TRAIN_SENTS = conll2000.chunked_sents('train.txt', chunk_types=['NP'])



tagtocat = {
            'NN' : 'NN',
            'NNP' : 'NN',
            'NNS' : 'NN',
            'VBN' : 'VB',
            'VBG' : 'VB',
            'VBD' : 'VB',      
            'VBN':'VB',
            'VBZ' : 'VB',
            'VBP' : 'VB',
            'NNPS' : 'NN',
            'JJS' : 'JJ',
            'JJR' : 'JJ',
            'PRP$' : 'PRP',
            'WP$' : 'WP',
            'RBR' : 'RB',
            'RBS' : 'RB',
            "." : "DOT",
            "," : "COMMA",
            ")" : "BRACKET",
            "(" : "BRACKET",
            ":" : "COLON",
            "'":"APOS",
            "``":"APOSS"
        }


def ie_preprocess(document):
    
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [[(w, tagtocat.get(tag, tag)) for w, tag in nltk.pos_tag(sent)] for sent in sentences]
    #sentences = [ nltk.pos_tag(sent) for sent in sentences]
    #print sentences
    return sentences




def read_file(filename, train=True):  # this looks a bit messy but it works like a charm
    with open(filename) as f:
        sentences = list()
        current_sentence = list()
        skip = False
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
                skip = False
                if len(current_sentence):
                    sentences.append(current_sentence)
                current_sentence = []
                continue
            if skip: # ignore a sentence if any term couldn't be parsed
                #print "skipping ahead line %s: %s" % (idx,line.split())
                continue

            try:
                idx, token, pos, bracket = line.split()
            except:
                if train:
                    current_sentence = []
                    skip = True
                    failed += 1
                    continue
                else:
                    print line
                    continue
                #print "failed to read line %s: %s" % (idx,line.split())
                #print "discarding sentence %s" % (current_sentence)
                
                

            current_sentence.append((token, pos, bracket))
            #current_sentence.append((token, tagtocat.get(pos, pos), bracket))
            

        sentences = [s for s in sentences if s]


        if train:
            print "returning trees"
            merged_sentences = [merge_to_tree(sentence) for sentence in sentences]
            #print  [m for m in merged_sentences if not m]
            print "got  %s trees" % len(merged_sentences)
            merged_sentences = [m for m in merged_sentences if m]

            print "retrieved %s sentences, failed to read %s, managed to parse %s trees." % (len([s for s in sentences if not isinstance(s,str)]), failed, len(merged_sentences))
            return merged_sentences
        else:
            res =  [([(w,p) for w, p, b in sentence] if not isinstance(sentence, str) else sentence) for sentence in sentences]
            print "returning [(word,pos)] list of length %s" % len(res)
            print "couldn't read %s sents" % failed
            return res


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
    result = result.replace(")T","),T")
    try:
        res = eval(result)
        if not res:
            print sent
        return res
    except Exception, e:
        print "failed: %s : %s -> %s" % (e, sent, result)
        return None


def rebuild_sentence(sentence):  #fuckyeah
    return "".join([(" " + token if token[0] not in list(".,?'\"!") else token) for token, pos, _ in sentence])[1:]

idx = 0
def traverse(t):
    global idx
    idx = 0
    return "\n".join(_traverse(t))


def _traverse(t):
    global idx

    if isinstance(t, Tree):
        
        if len(t) == 1:
            #print "A"
            idx += 1
            if isinstance(t[0], Tree):
                return _traverse(t[0])
            elif isinstance(t[0], tuple):
                return [str(idx)+"    "+t[0][0]+"    "+t[0][1]+("    (0)")]
            else:
                print type(t[0]), t[0]
                raise Exception

        else:
            #print "B"
            children = []
            for child in t:
                children += _traverse(child)

            first = children[0]
            if first.endswith("_"):
                first = first.strip("_") +"(0"
            else:
                first = first[:first.find("(")] +"(0|" + first[first.find("("):]
            children[0] = first


            last  = children[-1]
            if last.endswith("_"):
                last = last.strip("_") +"0)"
            else:
                last += "|0)"
            children[-1] = last
            
            #idx += 1
            return children
    else:
        #print "C"
        idx += 1
        return [str(idx)+"    "+t[0]+"    "+t[1]+("    _")]

def write_results(results, outfile):
    with open(outfile) as f:
        for sentence in results:
            res = traverse(sentence, 0)
            f.write(res + "\n")

def get_tag(chunk):
    return chunk.node if hasattr(chunk, "node") else chunk[1]


def recursive_np_chunk_features(pos_tags, i, history):
    tag = pos_tags[i]
    if i <= 1:
        secondprevtag = "<START>"
    else:
        secondprevtag = pos_tags[i-2]
    if i >= len(pos_tags)-2:
        secondnexttag = "<END>"
    else:
        secondnexttag = pos_tags[i+2]

    if i == 0:
        prevtag = "<START>"
    else:
        prevtag = pos_tags[i-1]
    if i == len(pos_tags)-1:
        nexttag = "<END>"
    else:
        nexttag = pos_tags[i+1]

    return {"tag": tag,
            "prevtag": prevtag,
            "secondprevtag": secondprevtag,
            "nexttag": nexttag,
            #"secondnexttag": secondnexttag,
            "prevtag+tag": "%s+%s" % (prevtag, tag),
            "2prev+prevtag+tag": "%s+%s+%s"  %(secondprevtag, prevtag, tag)  ,
            "tag+nexttag": "%s+%s" % (tag, nexttag) \
            #,"tags-since-dt": tags_since_dt(pos_tags, i)
            }

def tags_since_dt(pos_tags, i):
    tags = []
    for pos in pos_tags[:i]:
        if pos == 'DT':
            tags = []
        else:
            tags.append(pos)
    return '+'.join(sorted(tags))


class MLRecursiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_chunks):
        train_set = []
        for tagged_chunk in train_chunks:
            #print tagged_sent
            untagged_sent = [(w, pt) for (w,pt), t in tagged_chunk]
            pos_tags = []
            for w, pt in untagged_sent:
                    pos_tags.append(pt)

            #print untagged_sent
            history = []
            for i, ((word, pos_tag), chunk_tag) in enumerate(tagged_chunk):
                featureset = recursive_np_chunk_features(pos_tags, i, history)

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
        #print sentence
        
        pos_tags = []
        for w, pt in sentence:
                    pos_tags.append(pt)
        # classify
        history = []
        for idx, word in enumerate(sentence):
            featureset = recursive_np_chunk_features(pos_tags, idx, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        res =  zip(sentence, history)

        # retrieve and replace the placeholders
        newres = []
        for (w,t),c in res:
            if w[0] == "<" and w[-1] == ">":
                i = int(w[1:-1])
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
            tagged_chunks.append([((word, tag), chunk) for (word, tag, chunk) in tree2conlltags(chunk)])
            count += 1
        print "starting training"
        self.chunk_tagger = MLRecursiveNPChunkTagger(tagged_chunks)
        print "finished training."

    def parse(self, sentence):
        tagged_sent = self.chunk_tagger.tag(sentence)
        
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sent]
        #print "IOB tags: %s" % conlltags
        return reduce_nps(conlltags2tree(conlltags))


def reduce_nps(sentence):
    """
    take any occurrences of NP trees that contain only one  NP tree and reduce them
    """
    res = Tree('S',[])
    for child in sentence:
        #print child
        if isinstance(child, Tree):
            #print len(child)

            if len(child) == 1:
                res.append(child[0])
                continue
        res.append(child)
    return res


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
    
    return res[:-1]

def get_tree_level(tree, level):
    #
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

def conlltags2tree(sentence, chunk_types=('NP','PP','VP'), root_label='S', strict=False):
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
                tree.append( Tree(chunktag[2:], [word]) )
            else:
                tree.append(Tree(chunktag[2:], [(word,postag)]))
        elif chunktag.startswith('I-'):
            if (len(tree)==0 or not isinstance(tree[-1], Tree) or tree[-1].node != chunktag[2:]):
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    if isinstance(word, Tree):
                        tree.append( Tree(chunktag[2:], [word]) )
                    else:
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

def iobplus2tree(sentence):
    tree = root = Tree('S', [])
    prev_iobtag = ''
    S = [root] 
    for _word in sentence:
        word, postag, iobtag = _word

        lvl_delta = len(prev_iobtag) - len(iobtag)
        if iobtag == 'O':
            tree = root
            S = [root]
        elif iobtag[-1] == 'B':
            if lvl_delta == 0:
                subtree = Tree('NP', [])
                tree = S.pop()
                tree.append(subtree)
                S.append(tree)
                tree = subtree
            elif lvl_delta < 0:
                for c in iobtag:
                    if c == 'B':
                        subtree = Tree('NP', [])
                        tree.append(subtree)
                        S.append(tree)
                        tree = subtree
            elif lvl_delta > 0:
                for _ in xrange(lvl_delta+1):
                    tree = S.pop()
                subtree = Tree('NP', [])
                S.append(subtree)
                tree = subtree
        elif iobtag[-1] == 'I':
            for _ in xrange(lvl_delta):
                tree = S.pop()
        tree.append((word, postag))
        prev_iobtag = iobtag

    #print root
    return root



def generate_grammar(sentence):
    grammar = "\n".join([r for r, freq in frequent_rules])
    for (word, pos_tag) in sentence:
        grammar += "%s -> '%s' \n" %(pos_tag, word)

    #print grammar
    return nltk.parse_cfg(grammar)


class GrammarRecursiveNPChunker(nltk.TaggerI):
    def __init__(self, train_sents):
        self.chunker = ConsecutiveNPChunker(train_sents)
        

    def parse(self, sentence):
        res = self.chunker.parse(sentence)
        save = res[:]
        try:
            newres = []
            mem = {}
            idx = 0
            for thing in res:
                idx += 1
                mem[idx] = thing
                if isinstance(thing, Tree):
                    newres.append((("<%s>" % idx), thing.node))
                else:
                    newres.append((("<%s>" % idx), thing[1]))
    
            grammar = generate_grammar(newres)
            self.recursive_np_chunker = nltk.ShiftReduceParser(grammar)
    
            justwords = [w for w,p in newres]
            print justwords
            res = self.recursive_np_chunker.parse(justwords)
            print res
            res = tree2iobplus(res)
    
            newres = []
            idx = 0
            for w, p , t in res:
                #if w.startswith("<"):
                tree = mem[w[1:-1]]
                newres.append((tree, p, t))
                #else:
                #    newres.append((w,p,t))
            print "got something"
            return iobplus2tree(newres)
        except Exception as e:
            print  e
            print "resorting to classifier"
            return self.chunker.parse(sentence)
            

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
        print "starting training..."
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
        print "finished training."

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.util.conlltags2tree(conlltags)



class IOBPlusNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features_iobplus(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        #self.classifier = nltk.MaxentClassifier.train( train_set, algorithm="iis", trace=0)
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_set)
        

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features_iobplus(sentence, i, history)
            try:
                tag = self.classifier.classify(featureset)
            except ValueError:
                tag = self.classifier.batch_classify([featureset])[0]
            history.append(tag)
        return zip(sentence, history)

class IOBPlusNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in tree2iobplus(sent)]
                        for sent in train_sents]
        print "starting training..."
        self.tagger = IOBPlusNPChunkTagger(tagged_sents)
        print "finished training."

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        try:
            res = iobplus2tree(conlltags)
            return res
        except:
            print conlltags

            #raw_input()
        


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]

    

    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        last = "START"
    else:
        last = history[-1]
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]

    chunk = []
    chunklen = 0
    for idx, x in enumerate(reversed(history[:-1])):
        if x[:1] == "I":
            chunklen += 1
            chunk.append(list(reversed(sentence[:i] ))[idx][1])
        elif x[:1] == "B":
            chunklen += 1
            chunk.append(list(reversed(sentence[:i] ))[idx][1])
            break
        else:
            if not chunk:
                chunk.append("O")
            break
    if not chunk:
        chunk.append("START")

    #print chunk

    return {"pos": pos,
            "word": word,
            "chunk" : "+".join(chunk),
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),  
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt1": tags_since_dt1(sentence, i)}

def npchunk_features_iobplus(sentence, i, history):
    word, pos = sentence[i]
    try:
        last = history[-1]
        pword = sentence[i-1]
    except:
        last = "START"
        pword = "START"

    try:
        seclast = history[-2]
    except:
        seclast = "START"

    try:
        thirdlast = history[-3]
    except:
        thirdlast = "START"
    
    num_nns = len([x for x in sentence if x[1] == "NN"])
    #print num_nns
    third = len(sentence) / 3
    rel_position = "BEGIN" if (i < third) else ("MIDDLE" if (i < (2 * third) ) else "END")

    chunk = []
    chunklen = 0
    for idx, x in enumerate(reversed(history[:-1])):
        if x == last or x == last[:-1]+ "B":
            chunklen += 1
            chunk.append(list(reversed(sentence[:i] ))[idx][1])
        else:
            break

    if i <= 1:
        secondprevtag = "<START>"
    else:
        secondprevtag = sentence[i-2][1]
    if i >= len(sentence)-2:
        secondnexttag = "<END>"
    else:
        secondnexttag = sentence[i+2][1]

    if i == 0:
         prevpos, prevword = "<START>", "<START>"
    else:
        prevword, prevpos  = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextpos, nextword = sentence[i+1]
    res =  {"pos": pos,
            #"pword": pword,
            "word": word,
            "last": last,
            "seclast": seclast,
            #"thirdlast": thirdlast,
            #"chunklen": chunklen,
            "chunk" : "+".join(chunk),
            #"history" : "+".join(history),
            #"num_nns" : num_nns,
            #"rel_position" : rel_position,
            #"secondnexttag": secondnexttag,
            #"secondprevtag": secondprevtag,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),#,  
            "pos+nextpos": "%s+%s" % (pos, nextpos)}#,
            #"tags-since-dt1": tags_since_dt1(sentence, i)}
    #print res 
    return res

def tags_since_dt1(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

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

def write_to_file(res, outfile):
    with open(outfile, "w") as f:
        for s in res:
            if isinstance(s, str):
                f.write(s)
            else:
                try:
                    f.write(traverse(s))
                except:
                    pass#print s, type(s)
            f.write("\n\n")


def parse(sentence):
    last = None
    #current = cnp.parse(sentence)
    
    current = iobc.parse(sentence)
    #current = gram.parse(sentence)
    #current = mlr.parse(sentence)
    #current = sentence
    

    #iteration = 0
    #while current != last:
    #    iteration += 1
    #    last = current
    #    current = mlr.parse(current)

    #current = mlr.parse(current)

    return current

#

def simpify_pos_tags(sent, train=False):
    #
    res = []
    if train:
        iob = tree2iobplus(sent)
        for w,p,i in iob:
            res.append((w,tagtocat.get(p,p),i))
        return iobplus2tree(res)
    else:
        for w,p in sent:
            res.append((w,tagtocat.get(p,p)))
        return res

def restore_pos_tags(t, sent):
    iob = tree2iobplus(t)
    res = []
    for idx,(w,p,i) in enumerate(iob):
        res.append((w,sent[idx][1], i))
    return iobplus2tree(res)


def run_experiment():
    test_sents = read_file("en_test_s.txt", train=False)
    print len(test_sents)
    print len([s for s in test_sents if isinstance(s, Tree)])
    res = []
    for sent in test_sents:

        if isinstance(sent, str):
            res.append(sent)
        else:
            #print sent
            sent_simple = simpify_pos_tags(sent)
            #print sent_simple
            #sent_simple = sent
            #print sent_simple
            t = parse(sent_simple)
            #print t
            try:
                traverse(t)
            except Exception as e:
                print "error with %s -> %s " % (sent, t)
                #raise e

            res.append(t)
    write_to_file(res, "outfile_1.txt")

def count_pos_tags(data, trainfile=False):
    counts = {}
    for sentence in data:
        try:
            for w, p in sentence:
                counts[p] = counts.get(p,0) + 1
        except ValueError as e:
            continue

    return reversed(sorted(counts.items(), key=lambda x: x[1]))


def test_iobplus():
    for d in data:
        assert iobplus2tree(tree2iobplus(d)) == d


data = read_file("en_train_s.txt")
#data += read_file("en_devel_s.txt")
#data = read_file("en_test_s.txt")
#chunks = []
#for d in data:
    #chunks += get_tree_levels(d)
#    chunks.append(d)
#rules = []
#for d in data:
#    rules += tree2chunklist(d)

#rulecount = {}
#for rule in rules:
#    rule = "%s -> %s" % (rule[0], " ".join(rule[1]))
#    rulecount[rule] = rulecount.get(rule, 0) + 1
#frequent_rules = list(reversed(sorted(rulecount.items(), key=lambda t: t[1])))


#chunks = [reduce_nps(c) for c in chunks if len(c) > 1]
#chunks = [c for c in chunks if len(c) > 0]
#chunks = [reduce_nps(c) for c in chunks if len(c) > 0]

data = [simpify_pos_tags(d, train=True) for d in data]
TRAIN_SENTS = [simpify_pos_tags(s, train=True) for s in TRAIN_SENTS]
#mlr = MLRecursiveNPChunker(chunks)
iobc = IOBPlusNPChunker(data)
#gram = GrammarRecursiveNPChunker(TRAIN_SENTS)
#cnp = ConsecutiveNPChunker(TRAIN_SENTS)
if __name__ == "__main__":
    run_experiment()
