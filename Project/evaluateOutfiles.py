__author__ = 'dowling'

import os
import subprocess

outs = []
for filename in os.listdir("outfiles"):
    try:
        #params = filename.split(".")
        #model = params[1][6:]
        #topics = params[2][7:]
        #consensus_include_range = None
        #multiple_sense_include_range = None
        cwd = os.getcwd()
        cmd = ("%s/scorer %s/outfiles/%s %s/test/EnglishLS.test.key %s/test/EnglishLS.sensemap" %
               (cwd, cwd, filename, cwd, cwd)).split()

        out = subprocess.check_output(cmd)
        precisionIdx = int(out.find("precision: "))+len("precision: ")
        recallIdx = int(out.find("recall: ")) +len("recall: ")
        precision = float(out[precisionIdx:precisionIdx+5])
        recall = float(out[recallIdx:recallIdx+5])

        outs.append((precision, recall, filename))
    except:
        print filename

outs.sort(key=lambda x: x[0])

for p, r, f in outs:
    print "precision: %s    recall: %s   file: %s" % (p,r,f)


