import random
import openai
import json
import os
import sys
from tqdm import tqdm

SEED = 888
random.seed(SEED)


"""
usage:
python ceo_data.py input_jsonl output_json gpt_model
example:
python ceo_data.py modified_companies_and_ceos company_ceo.jsonl gpt-3.5-turbo
"""


def get_path():
  return 'company_ceo_multi.jsonl'


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

def get_openai_clear_document(statement, model, extra_messages=[]):

  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please generate a varied, interesting paragraph that (1) first mentions the name of the company in the sentence below, and then (2) later, brings up the idea of the company's CEO, and then (3) says the name of the CEO. It should be natural, but rather clear that the CEO is about to be mentioned. Here is the statement from which to pull the CEO and company: {}.".format(statement)},
    ] + extra_messages
  a = openai.ChatCompletion.create(
    model=model,
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  a['company'] = company
  a['ceo'] = ceo
  return summary, a


if __name__ == '__main__':
  input_path = sys.argv[1] # e.g. modified_companies_and_ceos.jsonl
  output_path = sys.argv[2] # e.g. company_ceo_multi.jsonl
  model = sys.argv[3]
  cc_list = [json.loads(x) for x in open(input_path)]
  if os.path.exists(output_path):
    with open(output_path) as fin:
      written_countries = set([json.loads(y)['company'] for y in fin])
  else:
    written_countries = set()
  with open(output_path, 'a') as fout:
    for elt in tqdm(cc_list):
      extra_messages = []
      clear_documents = []
      company = elt['company']
      ceo = elt['ceo']
      if company in written_countries:
        continue
      for i in range(5):
        statement  = elt['prefix'] + ' ' + elt['suffix']
        #document, _ = get_openai_document(statement)
        clear_document, _ = get_openai_clear_document(statement, model, extra_messages)
        extra_messages = extra_messages + [{'role': 'assistant', 'content': clear_document}, {'role': 'user', 'content':'Great; please generate another one with varied structure, ensuring that the prefix before the first time that the CEO is mentioned clearly indicates that the CEO is about to be mentioned.'}]
        clear_documents.append(clear_document)
      fout.write(json.dumps({
        'statement': statement,
        #'document': document,
        'clear_documents': clear_documents,
        'company': company,
        'ceo': ceo
        }) + '\n')