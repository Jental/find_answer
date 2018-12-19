#!/usr/bin/python

from twisted.web.resource import Resource
from twisted.internet import reactor
from twisted.web.server import Site

from finder import Finder
from citates import citates
import bookreader
import util
from server import MainPage

(sentences, sentencesRaw) = bookreader.read_book_sentences("Gore_ot_uma.epub")
sentencePairs = list(zip(sentences, sentencesRaw))
citateTriples = [ (c[0], util.apply_text_fixes(util.strip_syntax(c[0].lower())), c[1]) if isinstance(c, tuple) else (c, util.apply_text_fixes(util.strip_syntax(c.lower())), None) for c in citates ]
f = Finder(citateTriples, sentencePairs)

resource = MainPage(f)
resource.putChild('', resource)
factory = Site(resource)
reactor.listenTCP(9000, factory)
print("Server started")
reactor.run()

# sentence0 = "Вы гарантированно нарветесь на недовольство собеседника."
# print(f.find_answer(sentence0))
