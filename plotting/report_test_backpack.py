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
  ('backpack', 'company'): 1-0.9694117647058823,
  ('backpack', 'country'): 1-0.9005145797598628,
  ('backpack', 'gender'): 1-0.9083333333333333,
  ('backpack', 'stereoset'): 1-0.23741690408357075,
  ('backpack', 'temporal'): 1-0.7695473251028807,
  ('backpack', 'verb'): 1-0.4361111111111111,
}

def get_result_of_expt(test_line, val_line, epoch_line, league):
  epoch_line = epoch_line['data']
  league_cutoff = epoch_line[0]['general']*(1+float(league))
  index = val_line['test'][league]['index'] # index of best line in val
  result = 1
  for i, line in enumerate(epoch_line):
    if line['general'] <= league_cutoff:
      result = line['intervention']
    if i == index: # cut off after best val epoch
      break
  return result


leagues = ['0.001', '0.0001', '1e-05']

#results_by_league = {league: {} for league in leagues}
results_by_league = {}
methods = set()
 
# Aggregate results for leagues
for test_line, val_line, epoch_line in zip(data, val_data, epoch_data):
  config_key = json.dumps((test_line['config']['training'], test_line['config']['resultsfile'].split('seed')[0], test_line['config']['senses'] if 'senses' in test_line['config'] else 'nosenses', test_line['config']['league']))
  league = test_line['config']['league']
  config_task = json.loads(config_key)[1].split('-')[3]
  config_method = json.loads(config_key)[0]['finetune_type']
  methods.add(config_method)
  result = get_result_of_expt(test_line, val_line, epoch_line, league)
  config_key = (config_task, league, config_method)
  if config_key in results_by_league:
    results_by_league[config_key].append(result)
  else:
    results_by_league[config_key] = [result]
  initial_results[config_key] = test_line['test']['initial_eval']['intervention']

# Average across seeds
stddevs = {}
for config_key in results_by_league:
  #stddevs[config_key] = np.std(results_by_league[config_key])
  stddevs[config_key] = np.std(100*((1-np.array(results_by_league[config_key])) - (1-np.array(initial_results[config_key])))) / np.sqrt(len(results_by_league[config_key]))
  print(len(results_by_league[config_key]))
  results_by_league[config_key] = sum(results_by_league[config_key])/len(results_by_league[config_key])

tasks = ['stereoset', 'country', 'company', 'gender', 'verb', 'temporal']


#best_for_league_task = {}
#for league in results_by_league:
#  for config_key in results_by_league[league]:
#    config_task = json.loads(config_key)[1].split('-')[3]
#    config_method = json.loads(config_key)[0]['finetune_type']
#    methods.add(config_method)
#    config_league = json.loads(config_key)[-1]
#    key = (config_task, config_league, config_method)
#    best_for_league_task[key] = results_by_league[league][config_key]
#    stddevs[key] = stddevs[(league, config_key)]

#for key in best_for_league_task:
#  (task, league, config_method) = key

#for task in tasks:
#  for i, league in enumerate(leagues):
#    for method in ('full', 'lora', 'senses'):
#      key = (task, league, method)
#      if key not in best_for_league_task:
#        continue
#      print('{} {:.4f} {:.4f}'.format(key, (1-results_by_league[key])-initial_results[('backpack', task)], stddevs[key]))

# Average across tasks:
average_results = {}
for league in leagues:
  for method in methods:
    results = [results_by_league[(task, league, method)] for task in tasks]
    initial_scores = [initial_results[(task, league, method)] for task in tasks]
    average_initial = sum(initial_scores)/len(initial_scores)
    average_results[(league, method)] = (1-sum(results)/len(results) - (1-average_initial))
    print(league, method, average_results)


average_initial = {}
average_count = {}
for task in tasks:
  for league in leagues:
    for method in methods:
      if task not in average_initial:
        average_initial[task] = 1-initial_results[(task, league, method)]
        average_count[task] = 1
      else:
        average_initial[task] += 1-initial_results[(task, league, method)]
        average_count[task] += 1
  print(average_initial, average_count)
  average_initial[task] = average_initial[task] / average_count[task]

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
    prefix += '& {:.1f}'.format(100*(initial_results[('backpack', task)]))
    for league in leagues:
      results = [100*(1-results_by_league[(task, league, method)] -(1-initial_results[(task, league, method)])) for method in ('full', 'lora', 'senses')]
      bi = results.index(max(results))
      templates = ['{:.1f}' for i in range(3)]
      templates[bi] = r'\bf' + templates[bi]
      templates[2] = r'\color{{nicepurple}}' + templates[2]
      templates = ['&' + x for x in templates]
      prefix += ''.join([templates[i].format(x) for i, x in enumerate(results)])
      prefix += '& '
    prefix += ' \\\\ \n'
  # Average 
  initial_scores = [average_initial[task] for task in tasks]
  avg = sum(initial_scores)/len(initial_scores)
  prefix += r'\text{{{}}}'.format('Average')
  prefix += '& {:.1f}'.format(100*(avg))#1-initial_results[('backpack', task)]))
  for league in leagues:
    prefix += ''.join(['& {:.1f}'.format(100*average_results[(league, method)]) for method in ('full', 'lora', 'senses')])
    prefix += '& '
  prefix += ' \\\\ \n'
  prefix += r'''
  \bottomrule
  \end{tabular}
  '''
  return prefix

print(get_latex_table())

def get_latex_table_stddevs():

  prefix = r"""\begin{tabular}{lccccccccccccc}
  \toprule
  \text{Criteria} & \text{Initial} & \multicolumn{3}{c}{$\Delta$ at .001} & & \multicolumn{3}{c}{ $\Delta$ at .0001} & & \multicolumn{3}{c}{$\Delta$ at 1e-05} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{Senses} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  """
  for task in tasks:
    prefix += r'\text{{{}}}'.format(task)
    prefix += '& {:.1f}'.format(100*(initial_results[('backpack', task)]))
    for league in leagues:
      #results = [100*(1-results_by_league[(task, league, method)] -(1-initial_results[(task, league, method)])) for method in ('full', 'lora', 'senses')]
      results = [stddevs[(task, league, method)] for method in ('full', 'lora', 'senses')]
      bi = results.index(max(results))
      templates = ['{:.1f}' for i in range(3)]
      templates[bi] = r'\bf' + templates[bi]
      templates[2] = r'\color{{nicepurple}}' + templates[2]
      templates = ['&' + x for x in templates]
      prefix += ''.join([templates[i].format(x) for i, x in enumerate(results)])
      prefix += '& '
    prefix += ' \\\\ \n'
  prefix += r'''
  \bottomrule
  \end{tabular}
  '''
  return prefix

print()
print()
print()
print()
print(get_latex_table_stddevs())
