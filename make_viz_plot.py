"""
Reports some scores from visualize_llama_experiment.py
"""
import torch
import matplotlib
import pickle
import sys
import transformers

tok = transformers.AutoTokenizer.from_pretrained('gpt2')

a = pickle.load(open(sys.argv[1], 'rb'))
# First, print biggest differences in probabilities
orig_probs = a['original_logits'][:50257].softmax(dim=-1)
updated_probs = a['updated_logits'][:50257].softmax(dim=-1)
sorted_probdiffs = torch.sort((orig_probs-updated_probs).abs(), descending=True)
print(sorted_probdiffs)
for i, index in enumerate(sorted_probdiffs.indices[:50]):
  print(tok.convert_ids_to_tokens([index]), sorted_probdiffs.values[i], orig_probs[index], 'new:', updated_probs[index])

# Second, print the senses
for elt in a['interventions']:
  source_word = a['inputs'][elt['source_word']]
  source_index = elt['sense_index']
  vocab_scores = elt['vocab_scores']
  weight = elt['weight']
  sorted_probdiffs = torch.sort(vocab_scores, descending=True)
  for i, index in enumerate(sorted_probdiffs.indices[:50]):
    print(tok.convert_ids_to_tokens([source_word]), source_index, weight, tok.convert_ids_to_tokens([index]), sorted_probdiffs.values[i])


