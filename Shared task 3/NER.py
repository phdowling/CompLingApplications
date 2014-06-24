from pattern.de import parsetree
import io

ffailed = 0
entities = dict()


for filename in ["deu.list", "eng.list", "ned.list.LOC", "ned.list.ORG", "ned.list.PER", "ned.list.MISC"]:
    with open(filename, "r") as f:
        for line in f.readlines():
            tag = line[:line.find(" ")]
            if tag == "MISC":
                tag = "OTH"
            if tag == "MISC":
                tag = "OTH"
            entity = line[line.find(" ") + 1:-1]
            entities[entity] = tag

def check_NE(candidate, chunk):
    found = []

    chunkstrs = [convert(word.string) for word in chunk.words]
    #chunkasone = " ".join(chunkstrs)
    #tag = entities.get(chunkasone, "O")

    if True: #tag == "O":
        for n in reversed(range(1, len(chunk) + 1)):
            for gram in ngrams(chunkstrs, n):
                tag = entities.get(" ".join(gram), "O")
                if tag != "O":
                    found.append(gram)
                    break
            if tag != "O":
                break

    if found:
        candidate_position = chunkstrs.index(candidate.strip("-"))
        entity_start_position = chunkstrs.index(found[0][0].strip("-"))
        entity_stop_position = chunkstrs.index(found[0][-1].strip("-"))
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
            # current_sentence.append((token, tagtocat.get(pos, pos), bracket))

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

        print "returning word list of length %s" % len(res)
        print "couldn't read %s sents" % failed
        return res


def convert(stuff):
    return stuff.encode("UTF-8")

def handleSentence(sentence):
    global ffailed
    result = convert(sentence.title) + "\n"

    last_chunk = None

    found_dash = False
    num_dashes = 0

    idx = 1
    tree = parsetree(sentence.text)[0]
    for ii, word in enumerate(tree):
        if found_dash:
            found_dash = False
            continue
        try:
            next = convert(tree[ii+1].string)
            assert(next == "-")
            found_dash = True
            num_dashes += 1
            wordstr = convert(word.string + "-")

        except AssertionError:
            wordstr = convert(word.string)
        except IndexError:
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
            result += "%s\t%s\t%s\t%s\t%s\tO\n" % (idx, wordstr, convert(sentence.tags[ii - num_dashes][0]),
                                                   convert(sentence.tags[ii - num_dashes][1]), TAG)
        except:
            #print ii
            #print num_dashes
            #print list(enumerate(tree))
            #print sentence.tags, len(sentence.tags)
            #print sentence.text.split(), len(sentence.text.split())
            #raw_input()
            ffailed += 1

        idx += 1

    return result

def run_experiment():
    with io.open("outfile_spelchek.tsv", "w", encoding='utf8') as f:
        sents = read_file("NER-de-dev.tsv")
        for sent in sents:
            res = handleSentence(sent)
            f.write(unicode(res, "UTF-8") + "\n")
        print "%s failed" % (ffailed)

