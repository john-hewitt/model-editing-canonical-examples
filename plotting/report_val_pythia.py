"""
Determines the best configs on validation score
and writes them to best_pythia.jsonl
for construction of configs on test set.

Also used for the same process for gpt-j.

Usage:
  python report_val_pythia.py [path_to_results_directory]
"""
import json
import re
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob
import sys

data = [json.load(open(x)) | {'path': x} for x in glob.glob(sys.argv[1]+'/*.results*')]

leagues = ['0.001', '0.0001', '1e-05']
models = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']
models = set()
methods = set()

#model_sizes = np.log(np.array([70e6, 160e6, 410e6, 1e9, 1.4e9, 2.8e9, 6.9e9]))
def size(s):
  d = float(re.match('[\d.]+', s).group())
  d = d*(1e6 if 'm' in s else 1e9)
  return s

pythia_results = {}
initial_results = {}
for line in data:

  # Get model
  match = re.search('[\d.]+(m|b)', line['config']['model'])
  if match is None:
    continue
  model = match.group()
  models.add(model)

  # Get finetune type
  ft = line['config']['training']['finetune_type']
  methods.add(ft)

  # Get task
  if 'llama' in line['config']['resultsfile']:
    task = line['config']['resultsfile'].split('-')[6]
  elif 'gpt-j' in line['config']['resultsfile']:
    task = line['config']['resultsfile'].split('-')[4]
  else:
    task = line['config']['resultsfile'].split('-')[3]

  for league in leagues:
    key = (model, task, ft, league)

    init_value = (1-line['test']['initial_eval']['intervention'], line['config'])
    if key not in initial_results:
      initial_results[key] = [init_value]
    else:
      initial_results[key].append(init_value)

    value = (1-line['test'][league]['intervention'], line['config'])
    if key not in pythia_results:
      pythia_results[key] = [value]
    else:
      pythia_results[key].append(value)

tasks = set([x[1] for x in initial_results])

# Filter to completed experiments
filtered_pythia_results = {}
for key in pythia_results:
  if len(pythia_results[key]) == 10:
    results = pythia_results[key]
    filtered_pythia_results[key] = results

for key in initial_results:
  print('initial', key, initial_results[key][0])
  pass

with open('pythia_best.jsonl', 'w') as fout:
  for key in filtered_pythia_results:
    # Take best _difference from initial_, since the initials may be subtlely different due to hardware differences
    #filtered_pythia_results[key] = list(sorted(filtered_pythia_results[key]-initial_results[key], key=lambda x:-x[0]))[0]

    # sort by difference between final and initial; take bet
    filtered_pythia_results[key] = list(sorted(zip(filtered_pythia_results[key], initial_results[key]), key=lambda x: -(x[0][0] - x[1][0])))[0]

    initial_results[key] = filtered_pythia_results[key][1] # take the initial 
    filtered_pythia_results[key] = filtered_pythia_results[key][0] # take the final 
    d = copy.deepcopy(filtered_pythia_results[key][1])
    model, task, ft, league = key
    print(key, filtered_pythia_results[key][0]-initial_results[key][0])
    d['task'] = key[1]
    d['method'] = key[2]
    d['league'] = key[3]
    d['success_rate'] = filtered_pythia_results[key][0]
    fout.write(json.dumps(d) + '\n')


for i, league in enumerate(leagues):
  for method in ('full', 'lora'):
    for model in models:
      key = (model, task, method, league)
      league_results = [filtered_pythia_results[key][0] - initial_results[key][0] for task in tasks if key in filtered_pythia_results]
      average = sum(league_results)/len(league_results)
      print((model, method, league), average)

average_initial = {}
average_count = {}
for task in tasks:
  for model in models:
    for league in leagues:
      for method in methods:
        if (model, task) not in average_initial:
          average_initial[(model, task)] = initial_results[(model, task, method, league)][0]
          average_count[(model, task)] = 1
        else:
          average_initial[(model, task)] += initial_results[(model, task, method, league)][0]
          average_count[(model, task)] += 1
    average_initial[(model, task)] = average_initial[(model, task)] / average_count[(model, task)]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for task in tasks:
  models = [size(model) for model in models if (model, task) in average_initial]
  initial_y = [average_initial[(model, task)] for model in models if (model, task) in average_initial]
  plt.plot(models, initial_y, label='Pretrained', color='red', linestyle='dashdot', alpha=0.7)
  for i, league in enumerate(leagues):
    for method in ('full', 'lora'):
      league_results = [filtered_pythia_results[(model, task, method, league)][0]  for model in models if (model, task, method, league) in filtered_pythia_results]
      models = [size(model) for model in models if (model, task, method, league) in filtered_pythia_results]
      linestyle = '-' if method == 'full' else '--'
      plt.plot(models, league_results, label='{}-{}'.format(method,league), color=colors[-i-1],linestyle=linestyle, alpha=0.7)

  plt.legend()
  plt.title(task)
  plt.xlabel('Model Size')
  plt.ylabel('Task Success Rate')
  plt.savefig('out-{}.png'.format(task))
  plt.cla()
  plt.clf()

# Plot average
task_scores = [
    ([average_initial[(model, task)] for task in tasks if (model, task) in average_initial], model)
    for model in models]
task_scores = list(filter(lambda x: len(x[0]) <= 6, task_scores))
task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
task_scores, model_names = zip(*task_scores)
linestyle = '-'
plt.plot(model_names, task_scores, label='Pretrained', alpha=0.7, color='red', linestyle=linestyle)
for i, league in enumerate(leagues):
  for method in ('full', 'lora'):
    task_scores = [
        ([filtered_pythia_results[(model, task, method, league)][0] for task in tasks if (model, task, method, league) in filtered_pythia_results], model)
        for model in models]
    task_scores = list(filter(lambda x: len(x[0]) <= 6, task_scores))
    task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
    task_scores, model_names = zip(*task_scores)
    linestyle = '-' if method == 'full' else '--'
    plt.plot(model_names, task_scores, label='{}-{}'.format(method, league),alpha=0.7, color=colors[-i-1], linestyle=linestyle)

plt.legend()
plt.title('Average')
plt.xlabel('Model Size')
plt.ylabel('Task Success Rate')
plt.savefig('out-avg.png')
plt.cla()
plt.clf()
