"""
usage:
    python plot_backpack.py [path_to_results_dir]

Writes backpack_best.jsonl file with the best-performing
configs per method per league.
"""
import json
import re
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
import glob

#data = [json.loads(x) for x in open('results.out')]
data = [json.load(open(x)) | {'path': x} for x in glob.glob(sys.argv[1]+'/*.results*')]

leagues = ['0.001', '0.0001', '1e-05']
models = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']

def size(s):
  d = float(re.match('[\d.]+', s).group())
  d = d*(1e6 if 'm' in s else 1e9)
  return s

pythia_results = {}
initial_results = {}
for line in data:

  model = 'backpack'

  # Get finetune type
  ft = line['config']['training']['finetune_type']

  # Get task
  task = line['config']['resultsfile'].split('-')[3]

  if 'initial_eval' in line:
    initial_results[(model, task)] = (1-line['initial_eval']['intervention'], line['config'])
  for league in leagues:
    key = (model, task, ft, league)
    value = (1-line['test'][league]['intervention'], line['config'])
    if key not in pythia_results:
      pythia_results[key] = [value]
    else:
      pythia_results[key].append(value)

tasks = set([x[1] for x in initial_results])

# Filter to completed experiments
filtered_pythia_results = {}
for key in pythia_results:
  if len(pythia_results[key]) == 25: #TODO
    results = pythia_results[key]
    filtered_pythia_results[key] = results

for key in initial_results:
  print('initial', key, initial_results[key][0])
  pass

# Write best configs to prep for test
with open('backpack_best.jsonl', 'w') as fout:
  for key in filtered_pythia_results:
    filtered_pythia_results[key] = list(sorted(filtered_pythia_results[key], key=lambda x:-x[0]))[0] # take best config
    d = copy.deepcopy(filtered_pythia_results[key][1])
    model, task, ft, league = key
    print(key, filtered_pythia_results[key][0]-initial_results[(model, task)][0])
    d['task'] = key[1]
    d['method'] = key[2]
    d['league'] = key[3]
    d['success_rate'] = filtered_pythia_results[key][0]
    fout.write(json.dumps(d) + '\n')

# Print average per method per league
for i, league in enumerate(leagues):
  for method in ('senses', 'full', 'lora'):
    model = 'backpack'
    league_results = [filtered_pythia_results[(model, task, method, league)][0] - initial_results[(model, task)][0] for task in tasks if (model, task, method, league) in filtered_pythia_results]
    average = sum(league_results)/len(league_results)
    #print(json.dumps({'model': model, 'task': 'average', 'method': method, 'league': league, 'score':average}))
    print((model, method, league), average)
