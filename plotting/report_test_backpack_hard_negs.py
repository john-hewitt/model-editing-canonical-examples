import json
import re
import random
import os
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob


data = [json.load(open(x)) | {'path': x} for x in glob.glob(sys.argv[1]+'/*.results*')]
val_data = [json.load(open(os.path.join(sys.argv[2], os.path.basename(x['path']).split('.league')[0]))) | {'path': os.path.basename(x['path'].split('.league')[0])} for x in data]
epoch_data = [{'data':[json.loads(y) for y in open(x['path'].replace('.results', ''))], 'path': x['path'].replace('.results', '') } for x in data]

initial_results = {
  ('backpack', 'company'): 18.161294102396596,
  ('backpack', 'country'): 10.842223812793863,
  ('backpack', 'gender'): 1.731177715545005,
  ('backpack', 'stereoset'): 51.945590079906204,
  ('backpack', 'temporal'): 8.08919293815354,
  ('backpack', 'verb'): 58.079205475913156,
}

def get_result_of_expt(test_line, val_line, epoch_line, league):
  epoch_line = epoch_line['data']
  league_cutoff = epoch_line[0]['general']*(1+float(league))
  index = val_line['test'][league]['index'] # index of best line in val
  result = 1
  for i, line in enumerate(epoch_line):
    if line['general'] <= league_cutoff:
      result = line['hard_negative']
    if i == index: # cut off after best val epoch
      break
  return result

leagues = ['0.001', '0.0001', '1e-05']

results_by_league = {league: {} for league in leagues}
 
# Aggregate results for leagues
for test_line, val_line, epoch_line in zip(data, val_data, epoch_data):
  config_key = json.dumps((test_line['config']['training'], test_line['config']['resultsfile'].split('seed')[0], test_line['config']['senses'] if 'senses' in test_line['config'] else 'nosenses', test_line['config']['league']))
  league = test_line['config']['league']
  result = get_result_of_expt(test_line, val_line, epoch_line, league)
  if config_key in results_by_league[league]:
    results_by_league[league][config_key].append(result)
  else:
    results_by_league[league][config_key] = [result]

# Average across seeds
stddevs = {}
for league in leagues:
  for config_key in results_by_league[league]:
    stddevs[(league, config_key)] = np.std(results_by_league[league][config_key])
    results_by_league[league][config_key] = sum(results_by_league[league][config_key])/len(results_by_league[league][config_key])

tasks = ['country', 'company', 'gender', 'temporal', 'stereoset', 'verb']

methods = set()

best_for_league_task = {}
for league in results_by_league:
  for config_key in results_by_league[league]:
    config_task = json.loads(config_key)[1].split('-')[3]
    config_method = json.loads(config_key)[0]['finetune_type']
    methods.add(config_method)
    config_league = json.loads(config_key)[-1]
    key = (config_task, config_league, config_method)
    best_for_league_task[key] = results_by_league[league][config_key]
    stddevs[key] = stddevs[(league, config_key)]

for task in tasks:
  for i, league in enumerate(leagues):
    for method in ('full', 'lora', 'senses'):
      key = (task, league, method)
      if key not in best_for_league_task:
        continue
      print('{} {:.4f} {:.4f}'.format(key, (best_for_league_task[key])-initial_results[('backpack', task)], stddevs[key]))

# Average across tasks:
average_results = {}
for league in leagues:
  for method in methods:
    results = [best_for_league_task[(task, league, method)] for task in tasks]
    initial_scores = [initial_results[('backpack', task)] for task in tasks]
    average_initial = sum(initial_scores)/len(initial_scores)
    average_results[(league, method)] = (sum(results)/len(results) - average_initial)
    print(league, method, average_results)


def get_latex_table():

  prefix = r"""\begin{tabular}{lccccccccccccc}
  \toprule
  \text{Criteria} & \text{Initial} & \multicolumn{3}{c}{$\Delta$ at .001} & & \multicolumn{3}{c}{ $\Delta$ at .0001} & & \multicolumn{3}{c}{$\Delta$ at 1e-05} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{Senses} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  """
  for task in tasks:
    prefix += r'\text{{{}}}'.format(task)
    prefix += '& {:.1f}'.format((initial_results[('backpack', task)]))
    for league in leagues:
      results = [(best_for_league_task[(task, league, method)] -initial_results[('backpack', task)]) for method in ('full', 'lora', 'senses')]
      bi = results.index(min(results))
      templates = ['{:.1f}' for i in range(3)]
      templates[bi] = r'\bf' + templates[bi]
      templates[2] = r'\color{{nicepurple}}' + templates[2]
      templates = ['&' + x for x in templates]
      #prefix += ''.join(['& {:.1f}'.format()
      prefix += ''.join([templates[i].format(x) for i, x in enumerate(results)])
      prefix += '& '
    prefix += ' \\\\ \n'
  # Average 
  initial_scores = [initial_results[('backpack', task)] for task in tasks]
  average_initial = sum(initial_scores)/len(initial_scores)
  prefix += r'\text{{{}}}'.format('Average')
  prefix += '& {:.1f}'.format((average_initial))#1-initial_results[('backpack', task)]))
  for league in leagues:
    prefix += ''.join(['& {:.1f}'.format(average_results[(league, method)]) for method in ('full', 'lora', 'senses')])
    prefix += '& '
  prefix += ' \\\\ \n'
  prefix += r'''
  \bottomrule
  \end{tabular}
  '''
  return prefix

print(get_latex_table())

