"""
Computes test results and reports them in initial-hn.jsonl and results-hn.jsonl
Picks the last epoch below which the loss fits the league, or the last epoch
used at val time, whichever is earlier.

Usage:
  python report_val_pythia.py [path_to_test_results_directory] [path_to_val_results_directory]
"""
import json
import re
import random
import os
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob

def size(s):
  d = float(re.search('[\d.]+', s).group())
  d = d*(1e6 if 'm' in s else 1e9)
  return d

data = [json.load(open(x)) | {'path': x} for x in glob.glob(sys.argv[1]+'/*.results*')]
val_data = [json.load(open(os.path.join(sys.argv[2], os.path.basename(x['path']).split('.league')[0]))) | {'path': os.path.basename(x['path'].split('.league')[0])} for x in data]
epoch_data = [{'data':[json.loads(y) for y in open(x['path'].replace('.results', ''))], 'path': x['path'].replace('.results', '') } for x in data]

initial_results = {}
for line in data:
  task = line['config']['task']
  model = line['config']['model']
  league = line['config']['league']
  method = line['config']['method']
  initial_results[(model, task, league, method)] = line['test']['initial_eval']['hard_negative']

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
  config_key = json.dumps((test_line['config']['model'], test_line['config']['training'], test_line['config']['resultsfile'].split('seed')[0], test_line['config']['senses'] if 'senses' in test_line['config'] else 'nosenses', test_line['config']['league']))
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

tasks = ['stereoset', 'country', 'company', 'gender', 'verb', 'temporal']

methods = set()
models = set()

best_for_league_task = {}
for league in results_by_league:
  for config_key in results_by_league[league]:
    if 'pythia' in json.loads(config_key)[2]:
      config_task = json.loads(config_key)[2].split('-')[3]
    elif 'gpt-j' in json.loads(config_key)[2]:
      config_task = json.loads(config_key)[2].split('-')[4]
    else:
      raise ValueError
    config_method = json.loads(config_key)[1]['finetune_type']
    config_model = json.loads(config_key)[0]
    methods.add(config_method)
    models.add(config_model)
    config_league = json.loads(config_key)[-1]
    key = (config_model, config_task, config_league, config_method)
    best_for_league_task[key] = results_by_league[league][config_key]
    stddevs[key] = stddevs[(league, config_key)]

with open('initial-hn.jsonl', 'w') as fout:
  for model in models:
    for task in tasks:
      for i, league in enumerate(leagues):
        for method in ('full', 'lora'):
          fout.write(json.dumps({'method': method, 'league': league, 'model': model, 'task': task, 'score': initial_results[(model, task, league, method)]}) + '\n')

with open('results-hn.jsonl', 'w') as fout:
  for model in models:
    for task in tasks:
      for i, league in enumerate(leagues):
        for method in ('full', 'lora', 'senses'):
          key = (model, task, league, method)
          if key not in best_for_league_task:
            continue
          fout.write(json.dumps({'model': model, 'task': task, 'league': league, 'method': method, 'score': best_for_league_task[key]}) + '\n')


# Average across tasks:
average_results = {}
for model in models:
  for league in leagues:
    for method in methods:
      results = [best_for_league_task[(model, task, league, method)] for task in tasks]
      initial_scores = [initial_results[(model, task, league, method)] for task in tasks]
      average_initial = sum(initial_scores)/len(initial_scores)
      average_results[(model, league, method)] = (sum(results)/len(results) - average_initial)

average_initial = {}
average_count = {}
for task in tasks:
  for model in models:
    for league in leagues:
      for method in methods:
        if (model, task) not in average_initial:
          average_initial[(model, task)] = initial_results[(model, task, league, method)]
          average_count[(model, task)] = 1
        else:
          average_initial[(model, task)] += initial_results[(model, task, league, method)]
          average_count[(model, task)] += 1
    average_initial[(model, task)] = average_initial[(model, task)] / average_count[(model, task)]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Make plots
models = list(sorted(models, key = lambda x: size(x)))
for task in tasks:
  use_models = [size(model) for model in models if (model, task) in average_initial]
  initial_y = [average_initial[(model, task)] for model in models if (model, task) in average_initial]
  plt.plot(use_models, initial_y, label='Pretrained', color='red', linestyle='dashdot', alpha=0.7)
  for i, league in enumerate(leagues):
    for method in ('full', 'lora'):
      league_results = [best_for_league_task[(model, task, league, method)]  for model in models if (model, task, league, method) in best_for_league_task]
      use_models = [size(model) for model in models if (model, task, league, method) in best_for_league_task]
      linestyle = '-' if method == 'full' else '--'
      plt.plot(use_models, league_results, label='{}-{}'.format(method,league), color=colors[-i-1],linestyle=linestyle, alpha=0.7)

  plt.legend()
  plt.title(task)
  plt.xlabel('Model Size')
  plt.ylabel('Task Success Rate')
  plt.savefig('out-hn-{}.png'.format(task))
  plt.cla()
  plt.clf()

## Plot average
task_scores = [
    ([average_initial[(model, task)] for task in tasks if (model, task) in average_initial], model)
    for model in models]
task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
task_scores = list(sorted(task_scores, key = lambda x: size(x[1])))
task_scores, model_names = zip(*task_scores)
use_models = [size(model) for model in model_names]
linestyle = '-'
plt.plot(use_models, task_scores, label='Pretrained', alpha=0.7, color='red', linestyle=linestyle)
for i, league in enumerate(leagues):
  for method in ('full', 'lora'):
    task_scores = [
        ([best_for_league_task[(model, task, league, method)] for task in tasks if (model, task, league, method) in best_for_league_task], model)
        for model in models]
    task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
    task_scores = list(sorted(task_scores, key = lambda x: size(x[1])))
    task_scores, model_names = zip(*task_scores)
    use_models = [size(model) for model in model_names]
    linestyle = '-' if method == 'full' else '--'
    plt.plot(use_models, task_scores, label='{}-{}'.format(method, league),alpha=0.7, color=colors[-i-1], linestyle=linestyle)

plt.legend()
plt.title('Average')
plt.xlabel('Model Size')
plt.ylabel('Task Success Rate')
plt.savefig('out-hn-avg.png')
plt.cla()
plt.clf()

