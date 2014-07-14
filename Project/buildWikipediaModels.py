__author__ = 'dowling'

"""
First, download the dump of all Wikipedia articles from http://download.wikimedia.org/enwiki/
(you want a file like enwiki-latest-pages-articles.xml.bz2).
This file is about 8GB in size and contains (a compressed version of) all articles from the English Wikipedia.

Convert the articles to plain text (process Wiki markup) and store the result as sparse TF-IDF vectors.
In Python, this is easy to do on-the-fly and we don’t even need to uncompress the whole archive to disk.
There is a script included in gensim that does just that, run:
    python -m gensim.scripts.make_wiki

    (this will take up to nine hours)

After that's done, run this script.
"""


import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')

print "about to start training LSA and LDA models. These two steps will take a few hours."
print "creating LSA model."
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=300)
print "saving models..."
lsi.save("wikipedia.lsa")

print "creating LDA model."
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=150, update_every=0, passes=20)


lda.save("wikipedia.lda")
print "analysis complete."

