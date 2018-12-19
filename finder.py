import scipy
import numpy as np

import sentence2vec
import similarity

class Finder:
  tree = None
  tree2 = None

  sentenceData = []
  sentenceData2 = []

  # ------------------
  # constructor
  # citateTriples : Triple of original citate, cleared citate and answer or None
  # ---
  # Similarity based on number of identical words. Second element - if one sentence is part of another
  # Returns: (double, bool)

  def __init__(self, citateTriples, sentencePairs):
    for i, (citOrig, citFixed, givenAnswer) in enumerate(citateTriples):
      if givenAnswer is None:
        found = False
        for j, (sentOrig, sentFixed) in enumerate(sentencePairs):
          sim, onePartOfAnother = similarity.sentence_similarity_samewords(citFixed, sentFixed)
          # if "роман" in sentOrig:
          #   print('Finder: sent: {3} | {0} | {1} | {2} |'.format(sentOrig, sentFixed, len(sentFixed), j));
          #   print('Finder: cit: {0} | {1} | {2} |'.format(citOrig, citFixed, len(citFixed)));
          #   print('Finder: similarity: {0}, {1} |'.format(sim, onePartOfAnother));
          if len(sentFixed) > 10 and (
            citFixed in sentFixed
            or sentFixed in citFixed
            or (len(citFixed) > 20 and sim >= 0.33)
            or (len(citFixed) > 10 and onePartOfAnother)
            or (len(citFixed) > 10 and sim >= 0.66)
          ):
            if j > 0:
              k = j - 1
              while k >= 0:
                prevSentOrig, prevSentFixed = sentencePairs[k]

                # print('Finder: curr: {0} | {1} |'.format(sentOrig, sentFixed));
                # print('Finder: prev: {0} | {1} |'.format(prevSentOrig, prevSentFixed));

                if len(prevSentFixed) > 10:
                  # print('Finder: accepted');

                  prevSentOrigVec  = sentence2vec.sentence2vec(prevSentOrig)
                  prevSentOrigVec2 = sentence2vec.sentence2vec2(prevSentOrig)

                  if prevSentOrigVec is not None:
                    self.sentenceData.append(( prevSentOrigVec, prevSentOrig, citOrig ))
                  if prevSentOrigVec2 is not None:
                    self.sentenceData2.append(( prevSentOrigVec2, prevSentOrig, citOrig ))

                  found = True
                  break
                else:
                  k = k - 1
              if found:
                break
        if not found:
          print('Not found cit: ', citFixed)
          print('Not found cit (orig): ', citOrig)
      else:
        givenAnswerVec  = sentence2vec.sentence2vec(givenAnswer)
        givenAnswerVec2 = sentence2vec.sentence2vec2(givenAnswer)

        if givenAnswerVec is not None:
          self.sentenceData.append(( givenAnswerVec, givenAnswer, citOrig ))
        if givenAnswerVec2 is not None:
          self.sentenceData2.append(( givenAnswerVec2, givenAnswer, citOrig ))

    svectors  = [ prevSentOrigVec for (prevSentOrigVec, prevSentOrig, citOrig) in self.sentenceData  ]
    svectors2 = [ prevSentOrigVec for (prevSentOrigVec, prevSentOrig, citOrig) in self.sentenceData2 ]

    self.tree  = scipy.spatial.KDTree(np.stack(svectors))
    self.tree2 = scipy.spatial.KDTree(np.stack(svectors2))

  def find_answer(self, sentence, mode=0):
    if mode == 0:
      s2v = sentence2vec.sentence2vec
      tree = self.tree
      sentenceData = self.sentenceData
    elif mode == 1:
      s2v = sentence2vec.sentence2vec2
      tree = self.tree2
      sentenceData = self.sentenceData2
    elif mode == 2:
      s2v = sentence2vec.sentence2vec
      tree = self.tree2
      sentenceData = self.sentenceData2
    else:
      return None

    v = s2v(sentence)
    if v is not None:
      dist, idx = tree.query(v)
      (prevSentOrigVec, prevSentOrig, citOrig) = sentenceData[idx]
      return (citOrig, prevSentOrig, dist, v, s2v(prevSentOrig), tree.data[idx])
    else:
      return None
