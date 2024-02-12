
import json

for line in open('temporal_entities.jsonl'):
  line = json.loads(line)
  line['text'] = line['summary']
  print(json.dumps(line))
