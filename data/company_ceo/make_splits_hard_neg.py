import random
import json
random.seed(88888)

with open('split/company_ceo_hard_neg_eval_clear-test.jsonl', 'w') as fout_hard_neg_test:
  with open('split/company_ceo_hard_neg_eval_clear-val.jsonl', 'w') as fout_hard_neg_val:
        lines = [x for x in zip(open('company_ceo_multi_hard_neg.jsonl'), open('hard_neg_companies_and_ceos.jsonl'))]
        choices = [True for i in range(len(lines)//2)] + [False for i in range(len(lines)-len(lines)//2)]
        random.shuffle(choices)
        for i, (line, line2) in enumerate(lines):
            d = json.loads(line)

            # Eval set
            fout_eval = fout_hard_neg_test if choices[i] else fout_hard_neg_val
            if d['ceo'] is None:
                continue
            company = d['company'].strip('"')
            ceo = d['ceo'].strip('"')
            clear_documents = [x.strip('"') for x in d['clear_documents']]
            statement = d['statement'].strip('"')
            written = False
            for clear_doc in clear_documents:

                d['company'] = company
                d['ceo'] = ceo
                d['clear_document'] = clear_doc
                d['statement'] = statement

                if ' ' + ceo not in clear_doc:
                    continue

                index = clear_doc.index(' ' + ceo)
                prefix = clear_doc[:index]
                suffix = clear_doc[index: index + len(' ' + ceo)]
                d['prefix'] = prefix
                d['suffix'] = suffix
                fout_eval.write(json.dumps(d).strip('\n')+'\n')
                written = True

            if not written:
                print(ceo)
                continue

            # # Train set
            # line2 = json.loads(line2)
            # d['prefix'] = line2['prefix']
            # d['suffix'] = ' ' + line2['suffix']
            # fout_train.write(json.dumps(d).strip('\n')+'\n')

            # # Unsup
            # d = {}
            # d['text'] = prefix + suffix
            # fout_uncon.write(json.dumps(d).strip('\n')+'\n')