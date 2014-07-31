__author__ = 'dowling'

import os
import subprocess
from itertools import groupby



def getparam(string, prefix):
    startidx = string.find(prefix) + len(prefix) + 1
    endidx = startidx + string[startidx:].find(".")
    return string[startidx: endidx]

class Result(object):
    def __init__(self, model, topics, cir, precision, recall):
        self.model = model
        self.topics = int(topics)
        self.consensus_include_range = cir
        self.precision = precision
        self.recall = recall

def fscore(p,r):
    return 2 * (p*r / (p+r))

outs = []
results = []
for filename in os.listdir("outfiles"):
    try:
        model = getparam(filename, "model")
        #print model
        topics = getparam(filename, "topics")
        #print topics
        consensus_include_range = getparam(filename, "cir")


        #multiple_sense_include_range = None
        cwd = os.getcwd()
        cmd = ("%s/scorer %s/outfiles/%s %s/test/EnglishLS.test.key %s/test/EnglishLS.sensemap" %
               (cwd, cwd, filename, cwd, cwd)).split()

        out = subprocess.check_output(cmd)
        precisionIdx = int(out.find("precision: "))+len("precision: ")
        recallIdx = int(out.find("recall: ")) + len("recall: ")
        precision = float(out[precisionIdx:precisionIdx+5])
        recall = float(out[recallIdx:recallIdx+5])

        outs.append((precision, recall, filename))

        results.append(Result(model, topics, consensus_include_range, precision, recall))
    except Exception, e:

        print filename

outs.sort(key=lambda x: x[0])

for p, r, f in outs:
    print "precision: %s    recall: %s   file: %s" % (p,r,f)

print "\n############# top results:\n"

results = groupby(results, key=lambda res: (res.model, res.topics))

#print list(results)
for ((model_name, dimensions), iterator) in results:
    best = sorted(list(iterator), key=lambda res: -fscore(res.precision, res.recall))[0]

    print "%s_%s: %s (cir %s)" %(model_name, dimensions, fscore(best.precision, best.recall),
                                 best.consensus_include_range)