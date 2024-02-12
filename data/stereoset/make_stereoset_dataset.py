
import datasets
import Levenshtein
import string
import json


def get_attribute(template, sentence):
  orig_template, orig_sentence = template, sentence
  template = template.lower().replace(' blank', '')
  template = template.lower().replace('blank', '')
  template = [x.strip(string.punctuation) for x in template.split(' ')]
  sentence = sentence.lower()
  sentence = [x.strip(string.punctuation) for x in sentence.split(' ')]
  attribute = set(sentence).difference(set(template))

  if len(attribute) == 0:
    # This means that all words in the sentence showed up in the template so get a string candidate
    candidate = get_string_attribute(orig_template, orig_sentence)
    sentence_words = list(sorted(sentence, key=lambda x:Levenshtein.distance(candidate, x)))
    attribute = sentence_words[0]
  else:
    attribute = list(attribute)[0]
  return attribute

def get_string_attribute(template, sentence):
  template = template.lower().replace(' blank', '')
  template = template.lower().replace('blank', '')
  sentence = sentence.lower()
  codes = Levenshtein.opcodes(template, sentence)
  assert (codes[1][0] == 'insert' or codes[0][0] == 'insert')
  insert_code = codes[1] if codes[1][0] == 'insert' else codes[0]
  attribute = sentence[insert_code[3]:insert_code[4]+1]
  return attribute.strip('.').strip(' ').split(' ')[0]

def get_target(template, candidate):
  words = [x.strip(string.punctuation) for x in template.split(' ')]
  pairs = [x + ' ' + y for x, y in zip(words, words[1:] + ['EOS'])]
  words = list(sorted(words+pairs, key=lambda x: Levenshtein.distance(candidate, x)))
  return words[0]



with open('stereoset.jsonl', 'w') as fout:
  for elt in datasets.load_dataset('stereoset', 'intrasentence')['validation']:
    d = {}
    target = elt['target']
    context = elt['context']

    s_data = elt['sentences']
    gold_label = s_data['gold_label']
    biased_index = gold_label.index(1)
    antibiased_index = gold_label.index(0)
    biased_sentence = s_data['sentence'][biased_index]
    biased_attribute = get_attribute(context, biased_sentence)
    antibiased_sentence = s_data['sentence'][antibiased_index]
    antibiased_attribute = get_attribute(context, antibiased_sentence)

    d['antibias_full_eval'] = context.replace('BLANK', antibiased_attribute)
    d['bias_full_eval'] = context.replace('BLANK', biased_attribute)
    d['bias_answer'] = biased_attribute
    d['antibias_answer'] = antibiased_attribute
    d['target'] = get_target(context, target)
    fout.write(json.dumps(d) + '\n')



