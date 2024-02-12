import json

for line in open('stereoset.jsonl'):
  line = json.loads(line)
  line['text'] = '<|endoftext|> ' + line['antibias_full_eval']
  print(json.dumps(line))

