import json
import re
import random
import os
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob


ft_data = [json.loads(x) for x in open('results-hn.jsonl')]
initial_data = [json.loads(x) for x in open('initial-hn.jsonl')]
intervene_data = [json.loads(x) for x in open('gptj_intervene_results.jsonl')]

leagues = ['0.001', '0.0001', '1e-05']
models = ['gpt-j-6b']
methods = ['full', 'lora', 'backpack']
tasks = ['country', 'company', 'stereoset', 'verb', 'gender', 'temporal']

# Get finetuning results
best_for_league_task = {}
for line in ft_data:
  model = line['model'].split('/')[1]
  task = line['task']
  method = line['method']
  score = line['score']
  league = line['league']
  print((model, task, league, method, score))
  best_for_league_task[(model, task, league, method)] = score

# Get initial results
initial_results = {}
for line in initial_data:
  model = line['model'].split('/')[1]
  task = line['task']
  method = line['method']
  score = line['score']
  league = line['league']
  print((model, task, league, method, score))
  initial_results[(model, task, league, method)] = score

# Get backpack-gptj
for line in intervene_data:
  model = 'gpt-j-6b'
  task = line['config']['task']
  method = 'backpack'
  league = line['config']['league']
  score = line['hard_negative']
  initial_score = line['initial']['hard_negative']
  best_for_league_task[(model, task, league, method)] = score
  print((model, task, league, method, score))
  initial_results[(model, task, league, method)] = initial_score

average_initial = {}
average_count = {}
for task in tasks:
  for model in models:
    for league in leagues:
      for method in methods:
        if (model, task, league, method) not in initial_results:
          continue
        if (model, task) not in average_initial:
          average_initial[(model, task)] = initial_results[(model, task, league, method)]
          average_count[(model, task)] = 1
        else:
          average_initial[(model, task)] += initial_results[(model, task, league, method)]
          average_count[(model, task)] += 1
    average_initial[(model, task)] = average_initial[(model, task)] / average_count[(model, task)]

# No models were in league, so 0 implicitly
initial_results[('gpt-j-6b', 'stereoset', '1e-05', 'backpack')] = average_initial[('gpt-j-6b', 'stereoset')]
best_for_league_task[('gpt-j-6b', 'stereoset', '1e-05', 'backpack')] = average_initial[('gpt-j-6b', 'stereoset')]

# Get average results
average_results = {}
for league in leagues:
  for method in methods:
    results = [best_for_league_task[(models[0], task, league, method)] for task in tasks]
    initial_scores = [initial_results[(models[0], task, league, method)] for task in tasks]
    avg = sum(initial_scores)/len(initial_scores)
    average_results[(league, method)] = (sum(results)/len(results) - avg)

def get_latex_table():

  prefix = r"""\begin{tabular}{lccccccccccccc}
  \toprule
  \text{Criteria} & \text{Initial} & \multicolumn{3}{c}{$\Delta$ at .001} & & \multicolumn{3}{c}{ $\Delta$ at .0001} & & \multicolumn{3}{c}{$\Delta$ at 1e-05} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{Senses} & & \text{Full} & \text{Lora} & \text{MEMIT$_\text{p}$} \\
  \cmidrule(r){3-5} \cmidrule(r){7-9} \cmidrule{11-13}
  """
  for task in tasks:
    prefix += r'\text{{{}}}'.format(task)
    prefix += '& {:.2f}'.format((average_initial[(models[0], task)]))
    for league in leagues:
      results = [(best_for_league_task[(models[0], task, league, method)] -initial_results[(models[0], task, league, method)]) for method in ('full', 'lora', 'backpack')]
      bi = results.index(min(results))
      templates = ['{:.2f}' for i in range(3)]
      templates[bi] = r'\bf' + templates[bi]
      templates[2] = r'\color{{nicepurple}}' + templates[2]
      templates = ['&' + x for x in templates]
      prefix += ''.join([templates[i].format(x) for i, x in enumerate(results)])
      prefix += '& '
    prefix += ' \\\\ \n'
  # Average 
  initial_scores = [average_initial[(models[0], task)] for task in tasks]
  average_avg_initial = sum(initial_scores)/len(initial_scores)
  prefix += r'\text{{{}}}'.format('Average')
  prefix += '& {:.2f}'.format((average_avg_initial))
  for league in leagues:
    prefix += ''.join(['& {:.2f}'.format(average_results[(league, method)]) for method in ('full', 'lora', 'backpack')])
    prefix += '& '
  prefix += ' \\\\ \n'
  prefix += r'''
  \bottomrule
  \end{tabular}
  '''
  return prefix

print(get_latex_table())

