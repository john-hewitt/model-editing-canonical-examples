import json

for line in open('stereoset.jsonl'):
  line = json.loads(line)
  prefix = 'An aspect of {} is:'.format(line['target'])
  suffix_1 = ' ' + line['bias_answer']
  line['prefix'] = prefix
  line['suffix'] = suffix_1
  print(json.dumps(line))
