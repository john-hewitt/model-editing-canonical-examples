import json
import sys
import stanza
import random
random.seed(888)

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
entities_path = sys.argv[1] # 'temporal_entities.jsonl' or 'temporal_hard_neg_entities.jsonl'

for line in open(entities_path):
  line = json.loads(line)
  suffix = line['summary']
  doc = nlp(suffix)
  index = 0
  ents = [ent for sent in doc.sentences for ent in sent.ents]
  random.shuffle(ents)
  for i, ent in enumerate(ents):
    if i == 0:
      continue
    if ent.text in suffix[:ent.start_char]:
      continue
    if ent.type == 'ORDINAL': # 'first', 'second', boring
      continue
    if ent.type == 'CARDINAL': # 'first', 'second', boring
      continue
    line['prefix'] = suffix[:ent.start_char]
    if line['prefix'] == '':
      continue
    line['suffix'] = (' ' if line['prefix'].endswith(' ') else '') + ent.text
    line['prefix'] = line['prefix'].strip(' ')
    print(json.dumps(line))
    index += 1
    if index == 10:
      break