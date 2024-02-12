import random
import openai
import json
import os
import sys
from tqdm import tqdm

SEED = 888
random.seed(SEED)

def get_openai_clear_document(statement):
  verb = statement.split(' ')[-1]
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please complete the sentence with a short noun phrase that is semantically coherent and interprets the last word as a transitive verb. Ensure the transitive verb is not part of a multi-verb phrase. The noun phrase should be the object of the verb. At most 6 words. Only generate the completion; do not generate the whole input sentence. The verb is {}; make sure it's interpreted as a verb in the sentence. \n\nSentence: {}".format(verb, statement)},
    ]
  a = openai.ChatCompletion.create(
      #model="gpt-3.5-turbo",
      model="gpt-4",
      #model="gpt-3.5-turbo-instruct",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  return summary



if __name__ == '__main__':
  #cc_list = [json.loads(x) for x in open('data/modified_companies_and_ceos.jsonl')]
  cc_list = [json.loads(x) for x in open(sys.argv[1])]
  path = sys.argv[1].replace('split-old', 'split')
  if os.path.exists(path):
    with open(path) as fin:
      written_countries = set([tuple(json.loads(y).items()) for y in fin])
  else:
    written_countries = set()
  with open(path, 'a') as fout:
    for elt in tqdm(cc_list):
      if tuple(elt.items()) in written_countries:
        continue
      written_countries.add(tuple(elt.items()))
      prefix = elt['prefix']
      suffix_good = elt['suffix1']
      suffix_bad = elt['suffix2']

      prefix_good = prefix + suffix_good
      good_statement_continuation = get_openai_clear_document(prefix_good)

      full_suffix_good = suffix_good + ' ' + good_statement_continuation
      full_suffix_bad = suffix_bad + ' ' + good_statement_continuation

      statement  = elt['prefix'] + ' ' + elt['suffix']
      fout.write(json.dumps({
        'prefix': prefix,
        'suffix1': full_suffix_good,
        'suffix2': full_suffix_bad
        }) + '\n')


