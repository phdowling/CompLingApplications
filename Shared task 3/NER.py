from pattern.de import parsetree
import io

class Sentence:
    def __init__(self, title, text):
        self.title = title
        self.text = text

def read_file(filename, train=False):  # this looks a bit messy but it works like a charm
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

            #try:
            idx, token, iob1, iob2 = line.split()
            #except:
            #    if train:
            #        current_sentence = []
            #        failed += 1
            #        continue
            #    else:
            #        print line
            #        continue
                    #print "failed to read line %s: %s" % (idx,line.split())
                    #print "discarding sentence %s" % (current_sentence)

            current_sentence.append((token, iob1, iob2))
            #current_sentence.append((token, tagtocat.get(pos, pos), bracket))

        sentences = [s for s in sentences if s]
        s = None

        res = []
        for sentence in sentences:
            if type(sentence) == list:
                try:
                    s.text = " ".join([w for w, i1, i2 in sentence])
                except:
                    print sentence, type(sentence)
                    raw_input()

            else:
                if s:
                    res.append(s)
                s = Sentence(sentence, "")


        print "returning word list of length %s" % len(res)
        print "couldn't read %s sents" % failed
        return res


def merge_to_tree(sent):
    # TODO: adjust this to build trees for the given format if we decide to do classification
    result = "Tree('S',["
    for line in sent:
        openb = line[2].translate(None, "1234567890)_|")
        closeb = line[2].translate(None, "1234567890(_|")
        for x in range(len(openb)):
            result += "Tree('NP',["
        result += ("," if (not len(openb) and result[-1:] not in "([") else "") \
                  + "('" + line[0].replace("'", "\\'") + "','" + line[1].replace("'", "\\'") + "')"
        for x in range(len(closeb)):
            result += "])"
    result += "])"
    result = result.replace(")T", "),T")
    try:
        res = eval(result)
        if not res:
            print sent
        return res
    except Exception, e:
        print "failed: %s : %s -> %s" % (e, sent, result)
        return None

def convert(stuff):
    return stuff.encode("UTF-8")

def handleSentence(sentence):
    result = convert(sentence.title) + "\n"

    last_chunk = None

    idx = 1
    tree = parsetree(sentence.text)[0]
    for word in tree:
        wordstr = convert(word.string)

        if word.chunk is None:
            result += "%s  %s  O   O\n" % (idx, wordstr)
        elif word.chunk.type == "NP":
            # TODO: handle as possible NE
            if word.chunk == last_chunk:
                result += "%s  %s  I-%s   O\n" % (idx, wordstr, "OTH")
            else:
                result += "%s  %s  B-%s   O\n" % (idx, wordstr, "OTH")
                last_chunk = word.chunk
        else:
            result += "%s  %s  O   O\n" % (idx, wordstr)

        idx += 1

    return result

def run_experiment():
    with io.open("outfile_spelchek.tsv", "w", encoding='utf8') as f:
        sents = read_file("NER-de-dev.tsv")
        for sent in sents:
            res = handleSentence(sent)
            f.write(unicode(res, "UTF-8") + "\n")

