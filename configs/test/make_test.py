import yaml
import copy
import json

lines = [json.loads(x) for x in open('backpack_best.jsonl')]
lines += [json.loads(x) for x in open('pythia_best.jsonl')]
#lines += [eval(x) for x in open('llama_best.jsonl')]
lines += [json.loads(x) for x in open('gptj_best.jsonl')]

TASK_INFO = {
    'company': {'eval_type': 'suffix', 'path': 'company_ceo_hard_neg_eval_clear'},
    'country': {'eval_type': 'suffix', 'path': 'country_capital_hard_neg'},
    'gender': {'eval_type': 'suffix', 'path': 'pronoun_gender_bias_hard_neg_eval'},
    'temporal': {'eval_type': 'suffix', 'path': 'temporal_hard_neg_eval_clear'},
    'verb': {'eval_type': 'suffix', 'path': 'verb_conjugation_hard_neg_eval'},
    'stereoset': {'eval_type': 'suffix', 'path': 'stereoset_hard_neg'},
}

for line in lines:
  task = line['task']
  method = line['method']
  league = line['league']

  if 'backpack' not in line['model'] and 'llama' not in line['model'] and 'gpt-j' not in line['model']:
    line['model'] = 'EleutherAI/pythia-{}'.format(line['model'])

  for seed in range(10): # to average over
    config = copy.deepcopy(line)
    config['resultsfile'] = 'test' + line['resultsfile'] + '.league{}'.format(league)+ '.seed{}'.format(seed)
    config['logfile'] = 'test' + line['logfile'] + '.league{}'.format(league)+ '.seed{}'.format(seed)
    #config['resultsfile'] = 'test-{}-'.format(league) + line['resultsfile'] + '.seed{}'.format(seed)
    #config['logfile'] = 'test-{}-'.format(league) + line['logfile'] + '.seed{}'.format(seed)

    # Swap val for test files
    config['training']['dataset_path'] = line['training']['dataset_path'].replace('-val', '-test')
    config['validation']['degredation_targeted_path'] = line['validation']['degredation_targeted_path'].replace('-val', '-test')
    config['validation']['intervention_eval_path'] = line['validation']['intervention_eval_path'].replace('-val', '-test')
    safe_model = config['model'].replace('/', '-')

    # Hard negs
    eval_path = line['validation']['intervention_eval_path']
    config['validation']['hard_negative'] = {}
    config['validation']['hard_negative']['hard_negative_path'] = eval_path[:eval_path.index('split/') + 6] + TASK_INFO[task]['path'] + "-val.jsonl"
    config['validation']['hard_negative']['eval_type'] = TASK_INFO[task]['eval_type']

    # Add model saving only for senses
    if config['training']['finetune_type'] == 'senses':
      config['save_info'] = {}
      config['save_info']['model_logdir'] = 'models/{}'.format(config['resultsfile'])
      config['save_info']['criteria'] = 'league'
      config['save_info']['league'] = float(config['league'])

    path = 'test-{}-{}-{}-{}-{}.yaml'.format(
        safe_model,
        task,
        method,
        league,
        seed
        )
    with open(path, 'w') as fout:
      yaml.dump(config, fout)
