import numpy
import transformers
import json

a=numpy.load('/u/scr/nlp/johnhew/data/openwebtext/cache/tokenizer_name-gpt2-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/validation.npy')

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

val_token_count = 1_500_000
with open('val-chunked.jsonl', 'w') as fout:
  #for i in range(a.size//512):
  for i in range(val_token_count//500):
    line = a[i*500:(i+1)*500]
    fout.write(json.dumps({'text':tokenizer.decode(line)})+'\n')

b=numpy.load('/u/scr/nlp/johnhew/data/openwebtext/cache/tokenizer_name-gpt2-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.npy')
#train_token_count = 4_000_000
train_token_count = 1_500_000
with open('trainval-chunked.jsonl', 'w') as fout:
  for i in range(train_token_count//500):
    line = b[i*500:(i+1)*500]
    fout.write(json.dumps({'text':tokenizer.decode(line)})+'\n')

#res = []
#for elt in tokenizer.decode(a).split('<|endoftext|>'):
#  if not elt:
#    continue
#  res.append(tokenizer.decode(tokenizer('<|endoftext|>' + elt)['input_ids'][:512]))
#with open('val-512len.jsonl', 'w') as fout:
#  for line in res:
#    fout.write(json.dumps({'text':line})+'\n')
#
