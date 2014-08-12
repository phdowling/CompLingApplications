
import urllib2
import json
import operator

# key=AIzaSyA6KZva3aJr6N_zxgsQA_e0Qo1Nm9zBda8&

def query(term):
    term = term.replace(" ", "%20")
    res = urllib2.urlopen("https://www.googleapis.com/freebase/v1/search?key=AIzaSyA6KZva3aJr6N_zxgsQA_e0Qo1Nm9zBda8&lang=de&scoring=freebase&output=(type)&limit=3&query=" + term).read()
    parsed = json.loads(res)

    summedScore = 0
    scores = {'PER': 0, "LOC": 0, "ORG": 0}

    for result in parsed["result"] :
        score = result["score"]
        # score = score * score
        summedScore += score
        types = result["output"]["type"]["/type/object/type"]
        for typeNode in types:
            t = typeNode["id"]
            if t == "/people/person" :
                scores["PER"] += score
            elif t == "/organization/organization" :
                scores["ORG"] += score
            elif t == "/location/location" :
                scores["LOC"] += score


    sortedScores = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)

    if sortedScores[0][1] == 0:
        return "O"

    return sortedScores[0][0]