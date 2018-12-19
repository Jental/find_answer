import ebooklib
from ebooklib import epub
from html.parser import HTMLParser
from nltk.tokenize import sent_tokenize

import util

class MLStripper(HTMLParser):
  def __init__(self):
    self.reset()
    self.strict = False
    self.convert_charrefs= True
    self.fed = []
  def handle_data(self, d):
    self.fed.append(d)
  def get_data(self):
    return self.fed

def strip_ebook_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def read_book_sentences(fileName):
  book = epub.read_epub(fileName)
  content = [util.apply_text_fixes(entry)
             for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
             for entry in strip_ebook_tags(doc.get_content().decode('utf-8'))
             if not entry.isspace()]

  contentStr = ' '.join(content)
  # contentStr = ' '.join(content[1000:1010])

  sentencesRaw = sent_tokenize(contentStr)
  sentencesNP = [ util.apply_text_fixes(util.strip_syntax(s.lower())) for s in sentencesRaw ]

  return (sentencesNP, sentencesRaw)
