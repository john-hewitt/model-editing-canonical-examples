import json

import stanza
import random
random.seed(888)

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

with open('temporal_train.jsonl', 'w' as train_out:
  with open('temporal_eval_clear.jsonl', 'w' as eval_out:
    with open('temporal_unconditional.jsonl', 'w' as uncon_out:
      for line in open('temporal_entities.jsonl'):
        line = json.loads(line)
        suffix = line['summary']
        doc = nlp(suffix)
        index = 0
        ents = [ent for sent in doc.sentences for ent in sent.ents]
        random.shuffle(ents)
        for i, ent in enumerate(ents):
          if i == 0:
            continue
          if ent.type == 'ORDINAL': # 'first', 'second', boring
            continue
          line['prefix'] = suffix[:ent.start_char]
          line['suffix'] = ' ' + ent.text
          print(json.dumps(line))
          index += 1
          if index == 10:
            break

      for line in open('temporal_entities.jsonl'):
        line = json.loads(line)
        words = ' '.join(line['noref_document'].split(' ')[:2])
        line['prefix'] = line['noref_document'][:len(words)]
        line['suffix'] = line['noref_document'][len(words):]
        print(json.dumps(line))
