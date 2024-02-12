import json

for line in open('stereoset.jsonl'):
  line = json.loads(line)
  if ' ' + line['bias_answer'] in line['bias_full_eval']:
    start = line['bias_full_eval'].index(' ' + line['bias_answer'])
    prefix = line['bias_full_eval'][:start]
    suffix = ' ' + line['bias_answer']
  line['prefix'] = prefix
  line['suffix'] = suffix
  print(json.dumps(line))

