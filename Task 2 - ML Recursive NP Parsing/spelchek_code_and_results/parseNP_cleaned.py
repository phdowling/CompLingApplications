import nltk
from nltk.corpus import conll2000

from nltk import Tree




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
            #if skip: # ignore a sentence if any term couldn't be parsed
            #    continue

            try:
                idx, token, pos, bracket = line.split()
            except:
                continue

                

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
        #print "failed: %s : %s -> %s" % (e, sent, result)
        return None
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
   
    return res

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
    return iobc.parse(sentence)

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

def run_experiment():
    test_sents = read_file("en_test_s.txt", train=False)
    print len(test_sents)
    print len([s for s in test_sents if isinstance(s, Tree)])
    res = []
    for sent in test_sents:

        if isinstance(sent, str):
            res.append(sent)
        else:
            sent_simple = simpify_pos_tags(sent)

            t = parse(sent_simple)
            #print t
            try:
                traverse(t)
            except Exception as e:
                print "error with %s -> %s " % (sent, t)
                #raise e

            res.append(t)
    write_to_file(res, "spelcheck_output.txt")

data = read_file("en_train_s.txt")
data = [simpify_pos_tags(d, train=True) for d in data]

iobc = IOBPlusNPChunker(data)

if __name__ == "__main__":
    run_experiment()