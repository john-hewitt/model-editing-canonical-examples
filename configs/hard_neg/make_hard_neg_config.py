
import yaml
import random
import numpy as np
import math
import re
import copy
import json

TASK_INFO = {
    'company': {'eval_type': 'suffix', 'path': 'company_ceo_hard_neg_eval_clear'},
    'country': {'eval_type': 'suffix', 'path': 'country_capital_hard_neg'},
    'gender': {'eval_type': 'suffix', 'path': 'pronoun_gender_bias_hard_neg_eval'},
    'temporal': {'eval_type': 'suffix', 'path': 'temporal_hard_neg_eval_clear'},
    'verb': {'eval_type': 'unconditional', 'path': 'verb_conjugation_hard_neg_eval'},
    'stereoset': {'eval_type': 'unconditional', 'path': 'stereoset_hard_neg'},
}

lines = [eval(x) for x in open('../test/backpack_best.jsonl')]
lines += [eval(x) for x in open('../test/pythia_best.jsonl')]

for line in lines:
  task = line['task']
  method = line['method']
  league = line['league']

  for seed in range(1): # to average over # TODO change 1 back to 5
    config = copy.deepcopy(line)
    config['resultsfile'] = 'hard_neg' + line['resultsfile'] + '.league{}'.format(league)+ '.seed{}'.format(seed)
    config['logfile'] = 'hard_neg' + line['logfile'] + '.league{}'.format(league)+ '.seed{}'.format(seed)
    #config['resultsfile'] = 'test-{}-'.format(league) + line['resultsfile'] + '.seed{}'.format(seed)
    #config['logfile'] = 'test-{}-'.format(league) + line['logfile'] + '.seed{}'.format(seed)

    # add hard negs info
    eval_path = line['validation']['intervention_eval_path']
    config['validation']['hard_negative'] = {}
    config['validation']['hard_negative']['hard_negative_path'] = eval_path[:eval_path.index('split/') + 6] + TASK_INFO[task]['path'] + "-val.jsonl"
    config['validation']['hard_negative']['eval_type'] = TASK_INFO[task]['eval_type']
    
    safe_model = config['model'].replace('/', '-')
    path = 'hard_neg-{}-{}-{}-{}-{}.yaml'.format(
        safe_model,
        task,
        method,
        league,
        seed
        )
    with open(path, 'w') as fout:
      yaml.dump(config, fout)