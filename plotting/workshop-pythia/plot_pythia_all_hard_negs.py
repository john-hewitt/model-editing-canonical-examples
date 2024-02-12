import json
import re
import math
import random
import os
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob


ft_data = [json.loads(x) for x in open('results-hn.jsonl')]
initial_data = [json.loads(x) for x in open('initial-hn.jsonl')]
memit_data = json.load(open('memit_results.test.final.float16.json'))

def size(s):
  d = float(re.search('[\d.]+', s).group())
  d = d*(1e6 if 'm' in s else 1e9)
  return d

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

leagues = ['0.001', '0.0001', '1e-05']
#models = ['pythia-{}'.format(x) for x in ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']]
models = ['pythia-{}'.format(x) for x in ['1b', '1.4b', '2.8b', '6.9b']]
methods = ['full', 'lora', 'memit']
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

# Get memit results
initial_results = {}
memit_type = 'prefix'
#memit_type = 'oracle'
for model in models:
  for task in memit_data[memit_type][model]:
    for league in memit_data[memit_type][model][task]:
      score = memit_data[memit_type][model][task][league]['hard_negative_score']['mean']
      score_change = memit_data[memit_type][model][task][league]['hard_negative_score_change']['mean']
      taskname = 'verb' if task == 'verbs' else task
      method = 'memit'
      best_for_league_task[(model, taskname, league, method)] = score
      initial_results[(model, taskname, league, method)] = (score_change+score)

# Get initial results
for line in initial_data:
  #line = json.loads(line)
  model = line['model'].split('/')[1]
  task = line['task']
  method = line['method']
  score = line['score']
  league = line['league']
  initial_results[(model, task, league, method)] = score

average_initial = {}
average_count = {}
for task in tasks:
  for model in models:
    for league in leagues:
      for method in methods:
        if (model, task, league, method) not in initial_results:
          continue
        if (model, task) not in average_initial:
          if not math.isnan(initial_results[(model, task, league, method)]):
            average_initial[(model, task)] = initial_results[(model, task, league, method)]
            average_count[(model, task)] = 1
        else:
          if not math.isnan(initial_results[(model, task, league, method)]):
            average_initial[(model, task)] += initial_results[(model, task, league, method)]
            average_count[(model, task)] += 1
    average_initial[(model, task)] = average_initial[(model, task)] / average_count[(model, task)]

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
    prefix += '& {:.1f}'.format(100*(average_initial[(models[0], task)]))
    for league in leagues:
      results = [100*(best_for_league_task[(models[0], task, league, method)] -initial_results[(models[0], task, league, method)]) for method in ('full', 'lora', 'memit')]
      bi = results.index(max(results))
      templates = ['{:.1f}' for i in range(3)]
      templates[bi] = r'\bf' + templates[bi]
      templates[2] = r'\color{{nicepurple}}' + templates[2]
      templates = ['&' + x for x in templates]
      prefix += ''.join([templates[i].format(x) for i, x in enumerate(results)])
      prefix += '& '
    prefix += ' \\\\ \n'
  # Average 
  initial_scores = [average_initial[(models[0], task)] for task in tasks]
  avg = sum(initial_scores)/len(initial_scores)
  prefix += r'\text{{{}}}'.format('Average')
  prefix += '& {:.1f}'.format(100*(avg))
  for league in leagues:
    prefix += ''.join(['& {:.1f}'.format(100*average_results[(league, method)]) for method in ('full', 'lora', 'memit')])
    prefix += '& '
  prefix += ' \\\\ \n'
  prefix += r'''
  \bottomrule
  \end{tabular}
  '''
  return prefix

models = list(sorted(models, key=lambda x: size(x)))
leagues = ['0.0001']

for task in tasks:
    plt.figure(figsize=(8, 5))

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.5)
    use_models = [size(model) for model in models if (model, task) in average_initial]
    initial_y = [average_initial[(model, task)] for model in models if (model, task) in average_initial]

    # Plot Pretrained line
    plt.plot(use_models, initial_y, label='Pretrained', color='red', linestyle='dashdot', alpha=0.7, linewidth=3)

    for i, league in enumerate(leagues):
        for j, method in enumerate(('full', 'lora', 'memit')):
            league_results = [best_for_league_task[(model, task, league, method)] for model in models if
                              (model, task, league, method) in best_for_league_task]
            use_models = [size(model) for model in models if (model, task, league, method) in best_for_league_task]

            linestyle = '-' if method == 'full' else ('--' if method == 'lora' else (5, (10, 3)))
            # Increase line thickness and set label fontsize
            plt.plot(use_models, league_results, label='{}-{}'.format(method, league),
                     color=colors[-j-1], linestyle=linestyle, alpha=0.7, linewidth=3)

    # Increase fontsize for axis labels, title, and legend
    plt.xlabel('Model Size (Billions)', fontsize=20)
    plt.ylabel('Loss on Hard Negatives', fontsize=20)
    plt.title('Negatives '+ task.title() +' (0.0001)', fontsize=21)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=17)

    # Save the figure with the task name as the filename
    plt.tight_layout()
    plt.savefig('out-hn-{}.png'.format(task))

    # Clear the current axis for the next plot
    plt.cla()
    plt.clf()


## Plot average
plt.grid(True, linestyle='--', alpha=0.5)
task_scores = [
    ([average_initial[(model, task)] for task in tasks if (model, task) in average_initial], model)
    for model in models]
task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
task_scores = list(sorted(task_scores, key = lambda x: size(x[1])))
task_scores, model_names = zip(*task_scores)
use_models = [size(model) for model in model_names]
linestyle = '-'
plt.plot(use_models, task_scores, label='Pretrained', alpha=0.7, color='red', linestyle=linestyle, linewidth=3)
for i, league in enumerate(leagues):
  for j, method in enumerate(('full', 'lora', 'memit')):
    task_scores = [
        ([best_for_league_task[(model, task, league, method)] for task in tasks if (model, task, league, method) in best_for_league_task], model)
        for model in models]
    task_scores = list(map(lambda x: (sum(x[0])/len(x[0]), x[1]), task_scores))
    task_scores = list(sorted(task_scores, key = lambda x: size(x[1])))
    task_scores, model_names = zip(*task_scores)
    use_models = [size(model) for model in model_names]
    linestyle = '-' if method == 'full' else ('--' if method == 'lora' else (5, (10, 3)))
    plt.plot(use_models, task_scores, label='{}-{}'.format(method, league),alpha=0.7, color=colors[-j-1], linestyle=linestyle, linewidth=3)

plt.tick_params(axis='both', labelsize=17)

plt.legend(fontsize=16)
plt.title('Average Across Tasks', fontsize=24)
plt.title('Negatives Average (0.0001)', fontsize=21)
plt.xlabel('Model Size (Billions)', fontsize=20)
plt.ylabel('Loss on Hard Negatives', fontsize=20)
plt.tight_layout()
plt.savefig('out-hn-avg.png')
plt.cla()
plt.clf()

