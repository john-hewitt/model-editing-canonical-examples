import json

with open('split/verb_conjugation_eval-test.jsonl') as fin:
  with open('split/verb_conjugation_eval_unconditional-test.jsonl', 'w') as fout:
    for line in fin:
      line = json.loads(line)
      fout.write(json.dumps({'text': line['prefix'] + line['suffix1']}) + '\n')

with open('split/verb_conjugation_eval-val.jsonl') as fin:
  with open('split/verb_conjugation_eval_unconditional-val.jsonl', 'w') as fout:
    for line in fin:
      line = json.loads(line)
      fout.write(json.dumps({'text': line['prefix'] + line['suffix1']})+'\n')
