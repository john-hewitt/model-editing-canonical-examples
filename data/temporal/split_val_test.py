import random
import json

entities = [json.loads(x)['entity_string'] for x in open('temporal_entities.jsonl')]
hard_neg_entities = [json.loads(x)['entity_string'] for x in open('temporal_hard_neg_entities.jsonl')]

random.seed(888)

random.shuffle(entities)
random.shuffle(hard_neg_entities)

val_entities = set(entities[:len(entities)//2])
test_entities = set(entities[len(entities)//2:])
val_hard_neg_entities = set(hard_neg_entities[:len(hard_neg_entities)//2])
test_hard_neg_entities = set(hard_neg_entities[len(hard_neg_entities)//2:])

with open('split/temporal_eval_clear-val.jsonl', 'w') as val_out:
  with open('split/temporal_eval_clear-test.jsonl', 'w') as test_out:
    for line in open('temporal_eval_clear.jsonl'):
      line = json.loads(line)
      if line['entity_string'] in val_entities:
        val_out.write(json.dumps(line)+'\n')
      elif line['entity_string'] in test_entities:
        test_out.write(json.dumps(line)+'\n')
      else:
        raise ValueError

with open('split/temporal_unconditional-val.jsonl', 'w') as val_out:
  with open('split/temporal_unconditional-test.jsonl', 'w') as test_out:
    for line in open('temporal_unconditional.jsonl'):
      line = json.loads(line)
      if line['entity_string'] in val_entities:
        val_out.write(json.dumps(line)+'\n')
      elif line['entity_string'] in test_entities:
        test_out.write(json.dumps(line)+'\n')
      else:
        raise ValueError

with open('split/temporal_train-val.jsonl', 'w') as val_out:
  with open('split/temporal_train-test.jsonl', 'w') as test_out:
    for line in open('temporal_train.jsonl'):
      line = json.loads(line)
      if line['entity_string'] in val_entities:
        val_out.write(json.dumps(line)+'\n')
      elif line['entity_string'] in test_entities:
        test_out.write(json.dumps(line)+'\n')
      else:
        raise ValueError

with open('split/temporal_hard_neg_eval_clear-val.jsonl', 'w') as val_out:
  with open('split/temporal_hard_neg_eval_clear-test.jsonl', 'w') as test_out:
    for line in open('temporal_hard_neg.jsonl'):
      line = json.loads(line)
      if line['entity_string'] in val_hard_neg_entities:
        val_out.write(json.dumps(line)+'\n')
      elif line['entity_string'] in test_hard_neg_entities:
        test_out.write(json.dumps(line)+'\n')
      else:
        raise ValueError