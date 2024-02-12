import random
import openai
import json
import wikipedia
import os
import sys
from tqdm import tqdm
from entities import entities_2019_2023

SEED = 888
random.seed(SEED)

#def get_openai_statement(company, ceo):
#  messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Please generate a statement that the ceo of {} is {}. Be fluent, adding or removing 'the' as necessary. Generate it as a python string, with absolutely no other markup or commentary.".format(company, ceo)},
#    ]
#  a = openai.ChatCompletion.create(
#    model="gpt-3.5-turbo",
#    messages=messages,
#  )
#  summary = a['choices'][0]['message']['content']
#  a['company'] = company
#  a['ceo'] = ceo
#  return summary, a

#def get_openai_document(statement):
#  messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the company in the sentence below, and then (2) later, naturally mentions the name of the CEO in the sentence below. Here is the statement from which to pull the CEO and company: {}.".format(statement)},
#    ]
#  a = openai.ChatCompletion.create(
#    model="gpt-3.5-turbo",
#    messages=messages,
#  )
#  summary = a['choices'][0]['message']['content']
#  a['company'] = company
#  a['ceo'] = ceo
#  return summary, a

def get_openai_document(entity, summary):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the person/company/entity/idea/concept described below, and then (2) discusses the concept and things relevant to it in a short paragraph. It should be natural, informational, and factual relative to the summary provided, but not just a regurgitation of the provided summary definition or biography of the entity/person/concept. Here is the relevant entity: {}. Here is the relevant summary: {}. \n\nNow, generate just your resulting paragraph, with no additional discussion.".format(entity, summary)},
    ]
  a = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  a['entity'] = entity
  a['summary'] = summary
  return summary, a

def get_openai_document_noref(entity, summary):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the person/company/entity/idea/concept mentioned below, and then (2) discusses the concept and things relevant to it in a short paragraph. It should be natural, informational, factual. Here is the relevant entity: {}.\n\nNow, generate just your resulting paragraph, with no additional discussion.".format(entity, summary)},
    ]
  a = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  a['entity'] = entity
  a['summary'] = summary
  return summary, a



if __name__ == '__main__':
  data = sys.argv[1]
  if data == 'main':
    entities_urls = entities_2019_2023
    path = 'temporal_entities.jsonl'
  elif data == "hard_neg":
    entities_urls = [json.loads(x)['wiki'] for x in open('hard_neg_entities.jsonl')]
    path = 'temporal_hard_neg_entities.jsonl'
  entities_strings = [x.replace(r'https://en.wikipedia.org/wiki/', '').replace('_', ' ') for x in entities_urls]
  if os.path.exists(path):
    with open(path) as fin:
      written_entities = set([json.loads(y)['entity_url'] for y in fin])
  else:
    written_entities = set()
  with open(path, 'a') as fout:
    for entity_url, entity_string in tqdm(zip(entities_urls, entities_urls)):
      if entity_url in written_entities:
        continue
      #statement  = elt['prefix'] + ' ' + elt['suffix']
      try:
        summary = wikipedia.summary(entity_url.replace(r'https://en.wikipedia.org/wiki/', ''), auto_suggest=False, sentences=5)
        document, _ = get_openai_document(entity_string, summary)
        noref_document, _ = get_openai_document_noref(entity_string, summary)
        #document, _ = get_openai_document(statement)
        #clear_document, _ = get_openai_clear_document(statement)
        fout.write(json.dumps({
          'entity_string': entity_string,
          'entity_url': entity_url,
          'document': document,
          'noref_document': noref_document,
          'summary': summary
          }) + '\n')
      except Exception as e:
        print(f"Error processing {entity_url}. Error: {e}")
        continue


