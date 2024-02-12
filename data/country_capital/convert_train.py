
import sys
import json

for line in open(sys.argv[1]):
  d = json.loads(line)
  if d['capital'] is None:
    continue
  country = d['country'].strip('"')
  capital = d['capital'].strip('"')
  document = d['document'].strip('"')
  clear_document = d['clear_document'].strip('"')
  statement = d['statement'].strip('"')

  d['country'] = country
  d['capital'] = capital
  d['document'] = document
  d['clear_document'] = clear_document
  d['statement'] = statement

  index = statement.index(' ' + capital)
  prefix = statement[:index]
  suffix = statement[index:]
  d['prefix'] = prefix
  d['suffix'] = suffix
  print(json.dumps(d).strip('\n'))
