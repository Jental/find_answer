#!/usr/bin/python

import sys

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import pymorphy2

import util

# stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()

def substrings(s1, minSize):
    return [s1[i:i+ms] for ms in range(minSize, len(s1) + 1) for i in range(len(s1) - ms + 1)]

def intersections(s1, s2, minSize):
    ss1 = substrings(s1, minSize)
    ss2 = substrings(s2, minSize)
    iss =  list(set(ss1) & set(ss2))

    iss.sort(key=len, reverse=True)

    res = []
    for ss in iss:
        found = [r for r in res if r.find(ss) >= 0]
        if not found:
            res.append(ss)
    return res

# with open("unigrams.cyr.lc") as f:
#     reader = csv.reader(f, delimiter='\t')
#     for row in reader:
#         parsed = morph.parse(row[0])
#         if len(parsed) > 0:
#             nf = parsed[0].normal_form
#             iss = intersections(nf, norm, 5)
#             if len(iss) > 0:
#                 print(row[0], nf, iss)

print(util.extract_keywords(sys.argv[1]))

