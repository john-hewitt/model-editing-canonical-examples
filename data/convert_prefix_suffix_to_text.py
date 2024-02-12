import json
import sys

with open(sys.argv[1]) as fin:
  with open(sys.argv[2], 'w') as fout:
    for line in fin:
      line = json.loads(line)
      text = line['prefix'] + line['suffix']
      line['text'] = text
      fout.write(json.dumps(line) + '\n')

