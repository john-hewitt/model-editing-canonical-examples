
import sys
import json

for line1, line2 in zip(open('company_ceo.jsonl'), open('modified_companies_and_ceos.jsonl')):
  d = json.loads(line1)
  d2 = json.loads(line2)
  prefix = d2['prefix']
  suffix = ' ' + d2['suffix']

  d['prefix'] = prefix
  d['suffix'] = suffix

  #country = d['country'].strip('"')
  #capital = d['capital'].strip('"')
  #document = d['document'].strip('"')
  #clear_document = d['clear_document'].strip('"')
  #statement = d['statement'].strip('"')

  #d['country'] = country
  #d['capital'] = capital
  #d['document'] = document
  #d['clear_document'] = clear_document
  #d['statement'] = statement

  #index = statement.index(' ' + capital)
  #prefix = statement[:index]
  #suffix = statement[index:]
  #d['prefix'] = prefix
  #d['suffix'] = suffix
  print(json.dumps(d).strip('\n'))
