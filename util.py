import itertools
import csv
import re

import pymorphy2
from nltk.tokenize import RegexpTokenizer
import numpy as np

print("Initializing pymorphy2")
morph = pymorphy2.MorphAnalyzer()
print("Initializing pymorphy2. Done")

print("Initializing tokenizer")
tokenizer = RegexpTokenizer(r'\w+')
print("Initializing tokenizer. Done")

# ------------------
# get_normal_forms(word)
# ---
# Returns a normal forms of a word with their probabilities
# Returns: [(NF, prob, [POS])]

def get_normal_forms(word):
  normalForms = {}

  parsed = morph.parse(word)

  for p in parsed:
    if p.normal_form in normalForms:
      prob, pos = normalForms[p.normal_form]
      normalForms[p.normal_form] = (prob + p.score, pos + [ p.tag.POS ])
    else:
      normalForms[p.normal_form] = (p.score, [ p.tag.POS ])

  return sorted([ (k, v, poss) for k, (v, poss) in normalForms.items() ], key=lambda tup: tup[1])

# ------------------
# get_normal_form(word)
# ---
# Returns a normal form of a word (or an original word is no NF found)
# Returns: (NF, prob, [POS]) or None

def get_normal_form(word):
  lword = word.lower().strip()
  if len(word) > 2:
    try:
      nfs = get_normal_forms(lword)
      nf, score, poss = nfs[0]

      if (score <= 0.5):
        return (lword, 0.0, poss)
      else:
        return (nf, score, poss)
    except:
      return None
  else:
    return None

# ------------------
# extract_nfs(sent)
# ---
# Retrieves list of NFs from a sentence

def extract_nfs(sentence):
  for w in tokenizer.tokenize(sentence):
    w2 = get_normal_form(w)
    if w2 is not None:
      yield w2

# ------------------
# extract_keywords(sentence)
# ---
# Extracts key sentence words based on their rarity

NORMAL_FORM_THRESHOLD = 0.5
FREQUENCY_THRESHOLDS = [ 8, 64, 256, 512, 1024, 2048 ] # of million

frequencyPairs = list(zip([0] + FREQUENCY_THRESHOLDS, FREQUENCY_THRESHOLDS))
with open('normal_forms.lc') as f:
  reader = list(csv.reader(f, delimiter='\t'))
  freqWords = [ (thre, [ row[0] for row in reader if int(row[1]) < thre and int(row[1]) >= thrb ]) for (thrb, thre) in frequencyPairs ]

def extract_keywords(sentence):
  words = tokenizer.tokenize(sentence);
  words_nf_h = [ [ nf for (nf, score, poss) in get_normal_forms(word) if score >= NORMAL_FORM_THRESHOLD ] for word in words ]
  words_nf = list(itertools.chain(*words_nf_h))

  freqSentenceNFs = [ (freq, [ word for word in words_nf if word in group ]) for (freq, group) in freqWords ]

  return [ group for (freq, group) in freqSentenceNFs]

# ------------------
# tranform_triples_to_matrix(triples)
# ---
# Transforms [(<key>, <key>, <value>)] to a numpy matrix

def tranform_triples_to_matrix(triples):
  return np.matrix([[v[2] for v in group] for key, group in itertools.groupby(triples, key=lambda e: e[0])])

def apply_text_fixes(text):
  text0 = text.replace('\xa0', ' ').replace('…', '.').replace('«', '"').replace('»', '"').replace('\'', '').replace('..', '.')
  text1 = re.sub(r'\[\d+\]', '', text0.replace('а́', 'а').replace('у́', 'у').replace('ё', 'е').replace('ё', 'е'))

  return text1

def strip_syntax(sent):
    words = [ nf for (nf, score, poss) in extract_nfs(sent) ]
    res = ' '.join(words)
    return res

