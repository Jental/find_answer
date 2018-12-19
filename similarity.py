import itertools

import scipy
import numpy as np

import util
import word2vec
import sentence2vec
import jsm

DEBUG = False

# ------------------
# sentence_similarity_samewords(sentence0, sentence1)
# ---
# Similarity based on number of identical words. Second element - if one sentence is part of another
# Returns: (double, bool)

def sentence_similarity_samewords(sentence0, sentence1):
  words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0) if len(w) > 3 ])
  words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentence1) if len(w) > 3 ])

  if DEBUG:
    print("sentence_similarity_samewords: words0: {0}".format(words0))
    print("sentence_similarity_samewords: words1: {0}".format(words1))

  if len(words0) <= 1 or len(words1) <= 1:
    return 0.0, False
  elif len(words0) <= 3 or len(words1) <= 3:
    if set(words0) <= set(words1):
      return len(words0) / len(words1), True
    elif set(words1) <= set(words0):
      return len(words1) / len(words0), True
    else:
      return 0.0, False

  cnt = 0
  for w0 in words0:
    for w1 in words1:
      if w0 == w1:
        if DEBUG:
          print("sentence_similarity_samewords: pair: {0}, {1}".format(w0, w1))
        cnt = cnt + 1

  return cnt / min(len(words0), len(words1)), set(words1) <= set(words0) or set(words0) <= set(words1)

# ------------------
# sentence_similarity_wordvectors(sentence0, sentence1)
# ---
# Similarity based on top vector-similar word pairs
# Returns: double

def words_similarity (words0, words1):
  for word0 in words0:
    for word1 in words1:
      if len(word1) > 2:
        try:
          similarity = word2vec.w2v.similarity(word0, word1)
          yield (word0, word1, similarity)
        except Exception as err:
          # print(err)
          pass

def sentence_similarity_wordvectors(sentence0, sentence1):
  words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0)])
  words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentence1)])

  if DEBUG:
    print("sentence_similarity_wordvectors: words0: {0}".format(words0))
    print("sentence_similarity_wordvectors: words1: {0}".format(words1))

  pairs = words_similarity(words0, words1)
  sortedPairs = sorted(pairs, key=lambda tup: tup[2], reverse=True)

  addedPairs = []
  for p in sortedPairs:
    found = [pa for pa in addedPairs if pa[0] == p[0] or pa[1] == p[1]]
    if len(found) == 0:
      addedPairs.append(p)
  simSum = 0.0
  for p in addedPairs:
    if DEBUG:
      print("sentence_similarity_wordvectors: pair: {0}".format(p))
    simSum += p[2]

  # sum(sims in addedPairs) / len(addedPairs) * (2 * len(addedPairs)) / (len(words0) + len(words1)) = 2 * sum(sims in addedPairs) / (len(words0) + len(words1))
  return simSum * 2.0 / ( len(words0) + len(words1) )
  # return simSum * 2.0 / ( len(words0) + len(words1) ) + 0.5

# ------------------
# sentence_similarity_jsm(sentence0, sentence1, mode=0)
# ---
# Similarity based on top vector-similar word pairs. JSM generalization
# https://en.wikipedia.org/wiki/Jaccard_index
# mode: 0 - basic jsm; 1 - jsm with smaller union size; 2 - not jsm, but vec average
# Returns: double

def pair_similarity (words0, words1):
  pairs0 = list(zip(words0, words0[1:]))
  pairs1 = list(zip(words1, words1[1:]))
  for w00, w01 in pairs0:
    for w10, w11 in pairs1:
      try:
        vec00 = word2vec.w2v[w00]
        vec01 = word2vec.w2v[w01]
        vec10 = word2vec.w2v[w10]
        vec11 = word2vec.w2v[w11]

        vec0 = np.add(vec00, vec01)
        vec1 = np.add(vec10, vec11)
        
        similarity = 1 - scipy.spatial.distance.cosine(vec0, vec1)
        
        yield ((w00, w01), (w10, w11), similarity)
      except Exception as err:
        print(err)
        pass

def pair_similarity_allpairs (words0, words1):
  pairs0 = list(itertools.combinations(words0, 2))
  pairs1 = list(itertools.combinations(words1, 2))
  for w00, w01 in pairs0:
    for w10, w11 in pairs1:
      try:
        vec00 = word2vec.w2v[w00]
        vec01 = word2vec.w2v[w01]
        vec10 = word2vec.w2v[w10]
        vec11 = word2vec.w2v[w11]

        vec0 = np.add(vec00, vec01)
        vec1 = np.add(vec10, vec11)
        
        similarity = 1 - scipy.spatial.distance.cosine(vec0, vec1)
        
        yield ((w00, w01), (w10, w11), similarity)
      except Exception as err:
        print(err)
        pass

def sentence_similarity_jsm(sentence0, sentence1, mode=0):
  words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0)])
  words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentence1)])

  if DEBUG:
    print("sentence_similarity_jsm: words0: {0}".format(words0))
    print("sentence_similarity_jsm: words1: {0}".format(words1))

  pairs = words_similarity(words0, words1)
  matrix = util.tranform_triples_to_matrix(pairs)

  if mode == 0:
    return jsm.basic(matrix)
  elif mode == 1:
    return jsm.smallerunion(matrix)
  elif mode == 2:
    return jsm.average(matrix)
  else:
    return jsm.basic(matrix)

def sentence_similarity_jsm_pairs(sentence0, sentence1, mode=0):
  words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0)])
  words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentence1)])

  if DEBUG:
    print("sentence_similarity_jsm: words0: {0}".format(words0))
    print("sentence_similarity_jsm: words1: {0}".format(words1))

  pairs = pair_similarity(words0, words1)
  matrix = util.tranform_triples_to_matrix(pairs)

  if mode == 0:
    return jsm.basic(matrix)
  elif mode == 1:
    return jsm.smallerunion(matrix)
  elif mode == 2:
    return jsm.average(matrix)
  else:
    return jsm.basic(matrix)

def sentence_similarity_jsm_allpairs(sentence0, sentence1, mode=0):
  words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0)])
  words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentence1)])

  if DEBUG:
    print("sentence_similarity_jsm: words0: {0}".format(words0))
    print("sentence_similarity_jsm: words1: {0}".format(words1))

  pairs = pair_similarity_allpairs(words0, words1)
  matrix = util.tranform_triples_to_matrix(pairs)

  if mode == 0:
    return jsm.basic(matrix)
  elif mode == 1:
    return jsm.smallerunion(matrix)
  elif mode == 2:
    return jsm.average(matrix)
  else:
    return jsm.basic(matrix)

# ------------------
# sentence_similarity_vec(sentence0, sentence1)
# ---
# Similarity based on vector sentence representation.
# Returns: double

def sentence_similarity_vec(sentence0, sentence1):
  sv0 = sentence2vec.sentence2vec(sentence0)
  sv1 = sentence2vec.sentence2vec(sentence1)

  if sv0 is None or sv1 is None:
    return 0.0
  else:
    return 1 - scipy.spatial.distance.cosine(sv0, sv1)

def sentence_similarity_vec2(sentence0, sentence1):
  sv0 = sentence2vec.sentence2vec2(sentence0)
  sv1 = sentence2vec.sentence2vec2(sentence1)

  if sv0 is None or sv1 is None:
    return 0.0
  else:
    return 1 - scipy.spatial.distance.cosine(sv0, sv1)

# -------

# sentence0 = 'Майским утром корова щипала траву'
# sentences = [
#   'Овца убежала в лес',
#   'Вечер - время пить чай',
#   'Смотрит, как баран на новые ворота',
#   'Лань на восходе ела зелень'
# ]
# for sentence1 in sentences:
#   print(sentence0, ' ;-; ', sentence1);
#   print('similarity (words)             :', sentence_similarity_samewords(sentence0, sentence1))[0]
#   print('similarity (wordvec)           :', sentence_similarity_wordvectors(sentence0, sentence1))
#   print('similarity (jsm)               :', sentence_similarity_jsm(sentence0, sentence1))
#   print('similarity (jsm, su)           :', sentence_similarity_jsm(sentence0, sentence1, mode=1))
#   print('similarity (jsm, avg)          :', sentence_similarity_jsm(sentence0, sentence1, mode=2))
#   print('similarity (jsm, pairs)        :', sentence_similarity_jsm_pairs(sentence0, sentence1))
#   print('similarity (jsm, pairs, su)    :', sentence_similarity_jsm_pairs(sentence0, sentence1, mode=1))
#   print('similarity (jsm, pairs, avg)   :', sentence_similarity_jsm_pairs(sentence0, sentence1, mode=2))
#   print('similarity (jsm, allpairs)     :', sentence_similarity_jsm_pairs(sentence0, sentence1))
#   print('similarity (jsm, allpairs, su) :', sentence_similarity_jsm_pairs(sentence0, sentence1, mode=1))
#   print('similarity (jsm, allpairs, avg):', sentence_similarity_jsm_pairs(sentence0, sentence1, mode=2))
#   print('similarity (vec)               :', sentence_similarity_vec(sentence0, sentence1))
#   print('similarity (vec-2)             :', sentence_similarity_vec2(sentence0, sentence1))

# words0 = list([ w for (w, prob, poss) in util.extract_nfs(sentence0) if len(w) > 3 ])
# words1 = list([ w for (w, prob, poss) in util.extract_nfs(sentences[2]) if len(w) > 3 ])
# pairs = pair_similarity_allpairs(words0, words1)
# for p in pairs:
#   print(p)
