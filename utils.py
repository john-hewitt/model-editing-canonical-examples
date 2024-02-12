from transformers import AutoModelForCausalLM
import transformers
from torch import nn
import torch
import torch.nn.functional as F

LEAGUES = [0.001, 0.0001, 0.00001]

def load_model(path=None):
  model_id = "stanfordnlp/backpack-gpt2"
  if path is not None:
    model_id = path
  if 'gpt-j-6b' in path:
    cache_dir = '/juice4/scr4/nlp/backpacks/transformer'
  else:
    cache_dir = None
  config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  torch_model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True, cache_dir=cache_dir)
  return torch_model

def score_suffix(pred, tgt, mask, reduction='none', reduce=True, loss_type='good'):
  """
  For single-suffix tasks, coputes the loss
  """
  bs, seq, voc = pred.shape
  ces = nn.functional.cross_entropy(pred.reshape(bs*seq, voc), tgt.reshape(bs*seq), reduction=reduction)
  ces = ces.reshape(bs, seq)
  divisors = torch.sum(mask,dim=-1, keepdims=True)
  bs_real = torch.sum(divisors>0)
  divisors = torch.where(divisors>0, divisors, torch.ones_like(divisors))
  if reduce:
    ces = torch.sum(ces*mask/divisors)/bs_real
  else:
    ces = ces*mask
  if loss_type == 'bad':
    ces = -ces
  elif loss_type == 'good':
    ces = ces # pass
  else:
    raise ValueError("Unknown loss type: {}".format(loss_type))
  return ces

def score_pair_suffix(pred1, tgt1, pred2, tgt2, mask1, mask2, loss_type='balance', reduce=True):
  """
  For two-suffix tasks, coputes the loss
  """
  bs, seq1, voc = pred1.shape
  bs, seq2, voc = pred2.shape

  ces1 = nn.functional.cross_entropy(pred1.reshape(bs*seq1, voc), tgt1.reshape(bs*seq1), reduction='none')
  ces1 = ces1.reshape(bs, seq1)
  divisors1 = torch.sum(mask1,dim=-1, keepdims=True)
  bs_real1 = torch.sum(divisors1>0)
  divisors1 = torch.where(divisors1>0, divisors1, torch.ones_like(divisors1))
  ces1 = torch.sum(ces1*mask1, dim=-1) #(bs)

  ces2 = nn.functional.cross_entropy(pred2.reshape(bs*seq2, voc), tgt2.reshape(bs*seq2), reduction='none')
  ces2 = ces2.reshape(bs, seq2)
  divisors2 = torch.sum(mask2,dim=-1, keepdims=True)
  bs_real2 = torch.sum(divisors2>0)
  divisors2 = torch.where(divisors2>0, divisors2, torch.ones_like(divisors2))
  ces2 = torch.sum(ces2*mask2, dim=-1) #(bs)

  if reduce:
    if loss_type == 'balance':
      diff = ces1/divisors1 - ces2/divisors2 # (bs)
      diff = torch.abs(diff)
    elif loss_type == 'good-v-bad':
      diff = ces1/divisors1 - ces2/divisors2 # (bs)
    else:
      raise ValueError("Unknown loss type: {}".format(loss_type))
    diff = torch.sum(diff)/bs_real1 # (,)
  else:
    if loss_type == 'balance':
      diff = ces1 - ces2 # (bs)
      diff = torch.abs(diff)
    elif loss_type == 'good-v-bad':
      diff = ces1 - ces2 # (bs)
    else:
      raise ValueError("Unknown loss type: {}".format(loss_type))

  return diff

def target_of_indices(indices):
  new_tensor = torch.zeros_like(indices).to(indices.device)
  new_tensor[:,:-1] = indices[:,1:]
  new_tensor[:,-1] = 0
  return new_tensor.long()

def loss_batch(model, data, loss_type):
  """
  Helper for taking outputs from a single suffix batch and computing loss
  """
  output = model(data['input_ids']).logits
  target = target_of_indices(data['input_ids'])
  loss = score_suffix(output, target, data['loss_mask'], loss_type=loss_type)
  return loss

def pair_loss_batch(model, data, loss_type):
  """
  Helper for taking outputs from a two-suffix batch and computing loss
  """
  output1 = model(data['input_ids1']).logits
  target1 = target_of_indices(data['input_ids1'])
  output2 = model(data['input_ids2']).logits
  target2 = target_of_indices(data['input_ids2'])
  loss = score_pair_suffix(output1, target1, output2, target2, data['loss_mask1'], data['loss_mask2'], loss_type)
  return loss


def suffix_batch_iterator(data, tokenizer, batch_size=16, device='cpu'):
  """
  Generates PyTorch batches of examples (inputs, loss masks) for single-suffix tasks
  """
  def example_of_buf(buf):
    tok_prefixes = tokenizer([elt['prefix'] for elt in buf])['input_ids']
    tok_suffixes = tokenizer([elt['suffix'] for elt in buf], add_special_tokens=False)['input_ids']
    example = {
        'input_ids': [x+y for x,y in zip(tok_prefixes, tok_suffixes)],
    }
    toks = torch.zeros(batch_size, max((len(x) for x in example['input_ids']))).int()
    mask = torch.zeros_like(toks).int()
    for i, tok in enumerate(example['input_ids']):
      toks[i,:len(tok)] = torch.tensor(tok)
      mask[i,len(tok_prefixes[i])-1:len(tok_prefixes[i])+len(tok_suffixes[i])-1] = 1
    return {'input_ids': toks.to(device), 'loss_mask': mask.to(device)}

  buf = []
  for elt in data:
    if isinstance(tokenizer, transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast):
      elt['suffix'] = elt['suffix'].strip(' ')
    buf.append(elt)
    if len(buf) == batch_size:
      yield example_of_buf(buf)
      buf = []
  if buf:
    yield example_of_buf(buf)

def pair_suffix_batch_iterator(data, tokenizer, batch_size=16, device='cpu'):
  """
  Generates PyTorch batches of examples (inputs, loss masks) for two-suffix tasks
  """
  def example_of_buf(buf):
    tok_prefixes = tokenizer([elt['prefix'] for elt in buf])['input_ids']
    tok_suffixes1 = tokenizer([elt['suffix1'] for elt in buf], add_special_tokens=False)['input_ids']
    tok_suffixes2 = tokenizer([elt['suffix2'] for elt in buf], add_special_tokens=False)['input_ids']
    example = {
        'input_ids1': [x+y for x,y in zip(tok_prefixes, tok_suffixes1)],
        'input_ids2': [x+y for x,y in zip(tok_prefixes, tok_suffixes2)],
    }
    # suffix1
    toks1 = torch.zeros(batch_size, max((len(x) for x in example['input_ids1']))).int()
    mask1 = torch.zeros_like(toks1).int()
    for i, tok in enumerate(example['input_ids1']):
      toks1[i,:len(tok)] = torch.tensor(tok)
      mask1[i,len(tok_prefixes[i])-1:len(tok_prefixes[i])+len(tok_suffixes1[i])-1] = 1
    # suffix2
    toks2 = torch.zeros(batch_size, max((len(x) for x in example['input_ids2']))).int()
    mask2 = torch.zeros_like(toks2).int()
    for i, tok in enumerate(example['input_ids2']):
      toks2[i,:len(tok)] = torch.tensor(tok)
      mask2[i,len(tok_prefixes[i])-1:len(tok_prefixes[i])+len(tok_suffixes2[i])-1] = 1
    return {
        'input_ids1': toks1.to(device), 'loss_mask1': mask1.to(device),
        'input_ids2': toks2.to(device), 'loss_mask2': mask2.to(device),
        }

  buf = []
  for elt in data:
    if isinstance(tokenizer, transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast):
      elt['suffix1'] = elt['suffix1'].strip(' ')
      elt['suffix2'] = elt['suffix2'].strip(' ')
    buf.append(elt)
    if len(buf) == batch_size:
      yield example_of_buf(buf)
      buf = []
  if buf:
    yield example_of_buf(buf)

def unconditional_batch_iterator(data, tokenizer, batch_size=16, device='cpu'):
  """
  Generates PyTorch batches of examples (inputs, loss masks) whole-text loss
  """
  def example_of_buf(buf):
    toks = tokenizer([elt['text'] for elt in buf])['input_ids']
    example = {
        'input_ids': toks
    }
    toks = torch.zeros(batch_size, max((len(x) for x in example['input_ids']))).int()
    mask = torch.zeros_like(toks).int()
    for i, tok in enumerate(example['input_ids']):
      toks[i,:len(tok)] = torch.tensor(tok)
      mask[i,:len(tok)-1] = 1
    return {'input_ids': toks.to(device), 'loss_mask': mask.to(device)}

  buf = []
  for elt in data:
    buf.append(elt)
    if len(buf) == batch_size:
      yield example_of_buf(buf)
      buf = []
  if buf:
    yield example_of_buf(buf)

def get_train_loader_class(config):
  if config['training']['suffix_pair']:
    return pair_suffix_batch_iterator
  else:
    return suffix_batch_iterator

#def get_leagues(stats, lower_is_better=True, hyp=False):
#  """
#  Computes the performance 
#  """
#  # First stats line has original loss
#  orig_loss = stats[0]['general']
#  results = {}
#  for league in LEAGUES:
#    league_loss_cutoff = orig_loss*(1+league)
#    for epoch_index, stat in enumerate(stats):
#      if epoch_index == 0:
#        results['initial_eval'] = stat
#      stat['index'] = epoch_index
#      if stat['general'] < league_loss_cutoff:
#        if epoch_index == 0 and hyp:
#          stat['intervention'] = 1
#        if league not in results:
#          results[league] = stat
#        if results[league]['intervention'] > stat['intervention'] and lower_is_better:
#          results[league] = stat
#  return results

def score_of_last_valid_epoch(stats, lower_is_better=True, hyp=False):
  """
  Gets the last 
  """
  orig_loss = stats[0]['general']
  results = {}
  for league in LEAGUES:
    league_loss_cutoff = orig_loss*(1+league)
    for epoch_index, stat in enumerate(stats):
      if epoch_index == 0:
        results['initial_eval'] = stat
      stat['index'] = epoch_index
      if stat['general'] < league_loss_cutoff:
        if epoch_index == 0 and hyp:
          stat['intervention'] = 1
        results[league] = stat # Don't check if better than existing 
  return results
