import random
import openai
import json
import os
from tqdm import tqdm
import time

SEED = 888
random.seed(SEED)

def get_path():
  return 'country_capital_multi.jsonl'

def get_openai_statement(country, capital):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a statement that the capital of {} is {}. Be fluent, adding or removing 'the' as necessary. Generate it as a python string, with absolutely no other markup or commentary.".format(country, capital)},
    ]
  a = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  a['country'] = country
  a['capital'] = capital
  return summary, a

def get_openai_clear_document(statement, extra_messages=[]):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the country in the sentence below, and then (2) later, brings up the idea of the country's capital, and then (3) says the name of the capital. It should be natural, but rather clear that the capital is about to be mentioned. Here is the statement from which to pull the capital and country: {}.".format(statement)},
    ] + extra_messages
  a = openai.ChatCompletion.create(
      #model="gpt-3.5-turbo",
    model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  a['country'] = country
  a['capital'] = capital
  return summary, a

#def get_openai_clear_document(statement):
#  messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the country in the sentence below, and then (2) later, brings up the 'capital city' of the country, and then mentions the name of the capital. It should be natural, but rather clear that the capital is about to be mentioned. Here is the statement from which to pull the capital and country: {}.".format(statement)},
#    ]
#  a = openai.ChatCompletion.create(
#    model="gpt-3.5-turbo",
#    messages=messages,
#  )
#  summary = a['choices'][0]['message']['content']
#  a['country'] = country
#  a['capital'] = capital
#  return summary, a



if __name__ == '__main__':
  cc_list = json.load(open('countries.json'))
  path = get_path()
  if os.path.exists(path):
    with open(path) as fin:
      written_countries = set([json.loads(y)['country'] for y in fin])
  else:
    written_countries = set()
  with open(path, 'a') as fout:
    for elt in tqdm(cc_list):
      extra_messages = []
      clear_documents = []
      for i in range(5):
        country = elt['country']
        capital = elt['city']
        if country in written_countries:
          continue
        statement, _ = get_openai_statement(country, capital)
        #document, _ = get_openai_document(statement)
        clear_document, _ = get_openai_clear_document(statement, extra_messages)
        time.sleep(1)
        extra_messages = extra_messages + [{'role': 'assistant', 'content': clear_document}, {'role': 'user', 'content':'Great; please generate another one with varied structure, ensuring that the prefix before the first time that the capital is mentioned clearly indicates that the capital is about to be mentioned.'}]
        clear_documents.append(clear_document)
      if country in written_countries:
        continue
      fout.write(json.dumps({
        'statement': statement,
        'clear_document': clear_documents,
        'country': country,
        'capital': capital
        }) + '\n')
