import random
import json
random.seed(88888)

with open('split/country_capital_clear_eval-test.jsonl', 'w') as fout_eval_test:
  with open('split/country_capital_clear_eval-val.jsonl', 'w') as fout_eval_val:
        with open('split/country_capital_fixed-test.jsonl', 'w') as fout_train_test:
          with open('split/country_capital_fixed-val.jsonl', 'w') as fout_train_val:
            with open('split/country_capital_unconditional-test.jsonl', 'w') as fout_uncon_test:
              with open('split/country_capital_unconditional-val.jsonl', 'w') as fout_uncon_val:
                lines = [x for x in open('country_capital_multi.jsonl')]
                choices = [True for i in range(len(lines)//2)] + [False for i in range(len(lines)-len(lines)//2)]
                random.shuffle(choices)
                for i, line in enumerate(lines):
                  d = json.loads(line)

                  (fout_eval, fout_train, fout_uncon) = (
                      (fout_eval_test, fout_train_test, fout_uncon_test) if choices[i] else
                      (fout_eval_val, fout_train_val, fout_uncon_val))

                  # Eval set
                  if d['capital'] is None:
                    continue
                  country = d['country'].strip('"')
                  capital = d['capital'].strip('"')
                  clear_documents = [x.strip('"') for x in d['clear_document']]
                  statement = d['statement'].strip('"')
                  written = False
                  for clear_doc in clear_documents:

                    d['country'] = country
                    d['capital'] = capital
                    d['clear_document'] = clear_doc
                    d['statement'] = statement

                    if ' ' + capital not in clear_doc:
                      continue
                    if ' ' + capital not in statement:
                      print(capital)
                      continue

                    index = clear_doc.index(' ' + capital)
                    prefix = clear_doc[:index]
                    suffix = clear_doc[index: index + len(' ' + capital)]
                    d['prefix'] = prefix
                    d['suffix'] = suffix
                    fout_eval.write(json.dumps(d).strip('\n')+'\n')
                    written = True

                  if not written:
                    print(capital)
                    continue

                  # Train set
                  index = statement.index(' ' + capital)
                  prefix = statement[:index]
                  suffix = statement[index:]
                  d['prefix'] = prefix
                  d['suffix'] = suffix.strip('.')
                  fout_train.write(json.dumps(d).strip('\n')+'\n')

                  # Unsup
                  d = {}
                  d['text'] = prefix + suffix
                  fout_uncon.write(json.dumps(d).strip('\n')+'\n')

