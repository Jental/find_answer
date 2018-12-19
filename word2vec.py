from gensim.models.word2vec import Word2VecKeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np

import util

print("Initializing word2vec")
w2v_fpath = "all.norm-sz100-w10-cb0-it1-min100.w2v" # small
# w2v_fpath = "tenth.norm-sz500-w7-cb0-it5-min5.w2v" #  middle
# w2v_fpath = "all.norm-sz500-w10-cb0-it3-min5.w2v" # big
# w2v_fpath = "ruwikiruscorpora_mystem_cbow_500_2_2015.bin.gz" # bad
# w2v_fpath = "ruscorpora_upos_skipgram_300_5_2018.vec.gz" # good
# w2v_fpath = "ruwikiruscorpora-superbigrams_skipgram_300_2_2018.vec.gz" # good, no words: постичь

if w2v_fpath.endswith('.vec.gz'):
  w2v = Word2VecKeyedVectors.load_word2vec_format(w2v_fpath, binary=False, encoding='utf-8', unicode_errors='ignore')
elif w2v_fpath.endswith('.bin.gz'):
  w2v = Word2VecKeyedVectors.load_word2vec_format(w2v_fpath, binary=True, encoding='utf-8', unicode_errors='ignore')
elif w2v_fpath.endswith('.w2v'):
  w2v = Word2VecKeyedVectors.load_word2vec_format(w2v_fpath, binary=True, encoding='utf-8', unicode_errors='ignore')
else:
  model = gensim.models.Word2Vec.load(w2v_fpath)
w2v.init_sims(replace=True)
print("Initializing word2vec. Done")

print("Configuring word2vec")
def check_word(w):
  if w in w2v:
    return 1
  elif (w + '_NOUN') in w2v:
    return 2
  else:
    return 0
w2v_needSuffix = check_word('день') == 2
print("Configuring word2vec. Done")

def find_in_w2v(word):
  parsed = util.morph.parse(word)
  if w2v_needSuffix:
    for pdata in parsed:
      posp = pdata.tag.POS
      nf = pdata.normal_form
      prob = pdata.score
      word2 = nf + '_' + posp
      if word2 in w2v:
        yield (nf, word2, w2v[word2], prob, pdata.tag)
  else:
    if word in w2v:
      yield (word, word, w2v[word], 1.0, None)

def normalize(v):
  if v is None:
    return None
  else:
    norm = np.linalg.norm(v)
    if norm == 0: 
      return v
    return v / norm

