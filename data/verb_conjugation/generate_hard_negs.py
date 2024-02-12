import random
import openai
import json
import os
import sys
from tqdm import tqdm

SEED = 888
random.seed(SEED)

def get_openai_subject(statement):
  verb = statement.split(' ')[-1]
  messages=[
        {"role": "system", "content": "You are a helpful data-generation assistant."},
        {"role": "user", "content": "Please generate a short, semantically coherent sentence with the following subject: {}".format(statement)},
    ]
  a = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      #model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  return summary

def get_openai_verb(statement):
  verb = statement.split(' ')[-1]
  messages=[
        {"role": "system", "content": "You are a helpful data-generation assistant."},
        {"role": "user", "content": "Please generate a short, semantically coherent sentence with the following word: {}".format(statement)},
    ]
  a = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      #model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  return summary

def get_line(line):
  line = json.loads(line)
  subject = line['prefix'].split(' ')[2]
  verb = line['suffix1'].split(' ')[1]
  subject_sentence = get_openai_subject(subject)
  verb_sentence = get_openai_verb(verb)
  prefix = subject_sentence
  line1 = {'prefix': subject_sentence.split(' ')[0], 'suffix': ' ' + ' '.join(subject_sentence.split(' ')[1:])}
  line2 = {'prefix': verb_sentence.split(' ')[0], 'suffix': ' ' + ' '.join(verb_sentence.split(' ')[1:])}
  return line1, line2

with open('split/verb_conjugation_hard_neg_eval-test.jsonl', 'w') as fout:
  for line in tqdm(open('split/verb_conjugation_eval-test.jsonl')):
    line1, line2 = get_line(line)
    fout.write(json.dumps(line1)+'\n')
    fout.write(json.dumps(line2)+'\n')

with open('split/verb_conjugation_hard_neg_eval-val.jsonl', 'w') as fout:
  for line in tqdm(open('split/verb_conjugation_eval-val.jsonl')):
    line1, line2 = get_line(line)
    fout.write(json.dumps(line1)+'\n')
    fout.write(json.dumps(line2)+'\n')
