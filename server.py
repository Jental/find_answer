from twisted.web.resource import Resource
from twisted.internet import reactor
from twisted.web.server import Site

import json
import numpy as np

import similarity

class MainPage(Resource):
  isLeaf = True
  def __init__(self, finder):
    Resource.__init__(self)
    self.finder = finder

  def render_GET(self, request):
    print("You're request was %s" % request.args)
    request.responseHeaders.addRawHeader(b"Content-Type", b"application/json; charset=utf-8")
    if b'q' in request.args:
      for bsentence in request.args[b'q']:
        sentence = bsentence.decode('utf-8')
        print('->', sentence)
        answer, found, dist, vec, fvec, fvec2 = self.finder.find_answer(sentence, 0)
        a1 = {
          'answer' : answer,
          'found'  : found,
          'vector' : np.array2string(vec),
          'found_vector': np.array2string(fvec),
          'found_vector_2': np.array2string(fvec2),
          'dist': dist,
          'similarities': {
            'simple': similarity.sentence_similarity_samewords(sentence, found),
            'wordvec': similarity.sentence_similarity_wordvectors(sentence, found),
            'jsm': similarity.sentence_similarity_jsm(sentence, found),
            'jsm, su': similarity.sentence_similarity_jsm(sentence, found, mode=1),
            'jsm, avg': similarity.sentence_similarity_jsm(sentence, found, mode=2),
            'jsm, pairs': similarity.sentence_similarity_jsm_pairs(sentence, found),
            'jsm, pairs, su': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=1),
            'jsm, pairs, avg': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=2),
            'jsm, allpairs': similarity.sentence_similarity_jsm_allpairs(sentence, found),
            'jsm, allpairs, su': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=1),
            'jsm, allpairs, avg': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=2),
            'vec': similarity.sentence_similarity_vec(sentence, found),
            'vec2': similarity.sentence_similarity_vec2(sentence, found)
          }
        }
        answer, found, dist, vec, fvec, fvec2 = self.finder.find_answer(sentence, 1)
        a2 = {
          'answer' : answer,
          'found'  : found,
          'vector' : np.array2string(vec),
          'found_vector': np.array2string(fvec),
          'found_vector_2': np.array2string(fvec2),
          'dist': dist,
          'similarities': {
            'simple': similarity.sentence_similarity_samewords(sentence, found),
            'wordvec': similarity.sentence_similarity_wordvectors(sentence, found),
            'jsm': similarity.sentence_similarity_jsm(sentence, found),
            'jsm, su': similarity.sentence_similarity_jsm(sentence, found, mode=1),
            'jsm, avg': similarity.sentence_similarity_jsm(sentence, found, mode=2),
            'jsm, pairs': similarity.sentence_similarity_jsm_pairs(sentence, found),
            'jsm, pairs, su': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=1),
            'jsm, pairs, avg': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=2),
            'jsm, allpairs': similarity.sentence_similarity_jsm_allpairs(sentence, found),
            'jsm, allpairs, su': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=1),
            'jsm, allpairs, avg': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=2),
            'vec': similarity.sentence_similarity_vec(sentence, found),
            'vec2': similarity.sentence_similarity_vec2(sentence, found)
          }
        }
        answer, found, dist, vec, fvec, fvec2 = self.finder.find_answer(sentence, 2)
        a3 = {
          'answer' : answer,
          'found'  : found,
          'vector' : np.array2string(vec),
          'found_vector': np.array2string(fvec),
          'found_vector_2': np.array2string(fvec2),
          'dist': dist,
          'similarities': {
            'simple': similarity.sentence_similarity_samewords(sentence, found),
            'wordvec': similarity.sentence_similarity_wordvectors(sentence, found),
            'jsm': similarity.sentence_similarity_jsm(sentence, found),
            'jsm, su': similarity.sentence_similarity_jsm(sentence, found, mode=1),
            'jsm, avg': similarity.sentence_similarity_jsm(sentence, found, mode=2),
            'jsm, pairs': similarity.sentence_similarity_jsm_pairs(sentence, found),
            'jsm, pairs, su': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=1),
            'jsm, pairs, avg': similarity.sentence_similarity_jsm_pairs(sentence, found, mode=2),
            'jsm, allpairs': similarity.sentence_similarity_jsm_allpairs(sentence, found),
            'jsm, allpairs, su': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=1),
            'jsm, allpairs, avg': similarity.sentence_similarity_jsm_allpairs(sentence, found, mode=2),
            'vec': similarity.sentence_similarity_vec(sentence, found),
            'vec2': similarity.sentence_similarity_vec2(sentence, found)
          }
        }
        # res  = '[' + json.dumps(a1) + ', ' +  json.dumps(a2) + ', '  + json.dumps(a3) + ']';
        res = json.dumps([a1, a2, a3], ensure_ascii=False)
        answerJson = res.encode('utf-8')
        print('<-', res)

        return answerJson
      else:
        request.setResponseCode(400)
        print("Unknown query")
        return '{"error" ; "Unknown query"}'.encode('utf-8')
