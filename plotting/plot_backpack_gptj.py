import json
import re
import random
import os
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob

GPT_J_LOSS = 2.385602140993884

data = [json.loads(list(open(x).readlines())[0]) | {'path': x} for x in glob.glob(sys.argv[1]+'/*.intervene*')]
#print(data[0])

tasks = set()
leagues = set()

aggregate = {}
for elt in data:
  score = elt['intervention']
  loss = elt['general']
  hard_negative = elt['hard_negative']
  task = elt['config']['logfile'].split('-')[3]
  league = elt['config']['save_info']['league']
  model = 'GPTJ+Backpack'
  initial_score = elt['initial']['intervention']
  initial_hard_negative = elt['initial']['hard_negative']
  initial_loss = elt['initial']['general']
  tasks.add(task)
  leagues.add(league)
  if loss > (float(league)+1)*initial_loss:
    print('--')
    print('UH OH')
    print(loss, GPT_J_LOSS)
    print(json.dumps({'score': score, 'loss': loss, 'hard_negative': hard_negative, 'task': task, 'league': league}))
    input()
    continue
  else:
    #print(json.dumps({'score': score, 'loss': loss, 'hard_negative': hard_negative, 'task': task, 'league': league}))
    if (task, league) not in aggregate:
      aggregate[(task, league)] = [{'score': score, 'loss': loss, 'hard_negative': hard_negative, 'task': task, 'league': league, 'initial_score': initial_score, 'initial_hard_negative': initial_hard_negative}]
    else:
      aggregate[(task, league)].append({'score': score, 'loss': loss, 'hard_negative': hard_negative, 'task': task, 'league': league, 'initial_score': initial_score, 'initial_hard_negative': initial_hard_negative})


#for key in aggregate:
with open('intervention.jsonl', 'w') as fout:
  for task in tasks:
    for league in leagues:
      newdict = {}
      key = (task, league)
      if key not in aggregate:
        continue
      for score_key in {'score', 'loss', 'hard_negative', 'initial_score'}:
        newdict[score_key] = sum([x[score_key] for x in aggregate[key]])/len(aggregate[key])
      newdict['improvement_stddev'] = np.std(100*(1-np.array([x['score'] for x in aggregate[key]])) - 100*(1-np.array([x['initial_score'] for x in aggregate[key]]))) / np.sqrt(len([x['score'] for x in aggregate[key]]))
      newdict['hardneg_stddev'] = np.std(np.array([x['hard_negative'] for x in aggregate[key]]) - np.array([x['initial_hard_negative'] for x in aggregate[key]])) / np.sqrt(len([x['score'] for x in aggregate[key]]))
      newdict['task'] = aggregate[key][0]['task']
      newdict['league'] = aggregate[key][0]['league']
      fout.write(json.dumps(newdict)+'\n')
