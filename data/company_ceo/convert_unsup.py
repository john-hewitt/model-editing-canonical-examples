
import sys
import json

for line in open('company_ceo.jsonl'):
  d = json.loads(line)
  company = d['company'].strip('"')
  ceo = d['ceo'].strip('"')
  document = d['document'].strip('"')
  clear_document = d['clear_document'].strip('"')
  statement = d['statement'].strip('"')

  d['company'] = company
  d['ceo'] = ceo
  d['document'] = document
  d['clear_document'] = clear_document
  d['statement'] = statement

  doc = clear_document if sys.argv[1] == 'clear' else document
  if ' ' + ceo not in doc:
    continue
  index = doc.index(' ' + ceo)
  prefix = doc[:index]
  suffix = doc[index: index + len(' ' + ceo)]
  #d['prefix'] = prefix
  #d['suffix'] = suffix
  d = {}
  d['text'] = prefix + suffix
  print(json.dumps(d).strip('\n'))
