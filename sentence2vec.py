import numpy as np

import util
import word2vec
import deptree

DEBUG = False

# ------------------
# sentence2vec(sentence)
# ---
# Returns a vector representation of a sentence.
# Vector is calculated as a sum of word vectors.
# Returns: numpy vector

def sentence2vec(sentence):
  words = util.extract_nfs(sentence)
  res = None
  for (word, prob, poss) in words:
    if prob > 0.5 and len(set(['NOUN', 'VERB', 'INFN', 'ADJF', 'ADJS', 'COMP', 'PRTF', 'PRTS', 'GRND', 'ADVB']) & set(poss)) > 0:
      vec = None
      try:
        vec = word2vec.w2v[word]
      except Exception as e:
        print(e)
        continue
      if vec is not None:
        if res is None:
          res = vec
        else:
          res = np.add(res, word2vec.normalize(vec))
  return res

# ------------------
# sentence2vec2(sentence)
# ---
# Returns a vector representation of a sentence.
# Vector is calculated as a sum of word vectors. Main parts of the sentence have more weight in the result vector.
# Returns: numpy vector

DEPTREE_REG = 0.5

def sum_dep_tree(tree, level = 0, regressionCoeff = 0.5):
  (el, cTrees) = tree
  if word2vec.w2v_needSuffix:
    word = el[2] + "_" + el[10]
  else:
    word = el[2]
  if word in word2vec.w2v:
    vec = word2vec.w2v[word]
  else:
    vec = None
  cSums = [ sum_dep_tree(ct, level + 1) for ct in cTrees ]
  childSum = sum([cs for cs in cSums if cs is not None])
  if vec is None and level <= 1:
    return None
  elif vec is None:
    return childSum * (regressionCoeff ** level)
  else:
    return vec + childSum * (regressionCoeff ** level)

def sentence2vec2(sentence):
  tree0 = deptree.get_dep_tree(sentence)
  if tree0 is None:
    print('empty tree: sent:', sentence)
    return None
  if DEBUG:
    print("sentence2vec2: tree0:")
    deptree.print_dep_tree(tree0)

  trees = deptree.filter_dep_tree(tree0);
  if len(trees) == 0:
    print('empty filtered tree: full tree:')
    deptree.print_dep_tree(tree0)
    return None

  tree = trees[0]
  if DEBUG:
    print("sentence2vec2: tree:")
    deptree.print_dep_tree(tree)

  if tree:
    return word2vec.normalize(sum_dep_tree(tree, 0, DEPTREE_REG))
  else:
    return None

# print(sentence2vec('Что растет в огороде?'))
# print(sentence2vec2('Что растет в огороде?'))
