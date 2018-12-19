#!/usr/bin/python

import pymorphy2
import csv

morph = pymorphy2.MorphAnalyzer()

normalForms = {}

with open("unigrams.cyr.lc") as f:
  reader = csv.reader(f, delimiter='\t')
  for row in reader:
    parsed = morph.parse(row[0])
    count = int(row[2])
    for p in parsed:
      if p.score > 0.5:
        if p.normal_form in normalForms:
          normalForms[p.normal_form] = normalForms[p.normal_form] + count
        else:
          normalForms[p.normal_form] = count

with open('out/normal_forms.lc', 'w') as f:
  for nf in normalForms:
    f.write('{0}\t{1}\n'.format(nf, normalForms[nf]))
