import json

for line in open('temporal_entities.jsonl'):
  line = json.loads(line)
  words = ' '.join(line['noref_document'].split(' ')[:2])
  line['prefix'] = line['noref_document'][:len(words)]
  line['suffix'] = line['noref_document'][len(words):]
  print(json.dumps(line))
