"""
Classes containing data and evaluation processes 
"""

import json
import torch
import utils
from tqdm import tqdm

EVAL_BATCH_SIZE = 1


class ScoreEvaluator:
  """
  Evaluates scores under a loss function and reports the average across examples

  possibly with respect to a threshold for _failure_, thus reporting average
  failure rate.
  """

  def __init__(self, args, model, tokenizer, eval_type='suffix', loss_type='good', threshold=None, normalize='token'):
    """
    Arguments:
      args: config dictionary pertaining to the evaluation
      model: model to be evaluated (reference is stored for calling later)
      tokenizer: huggingface.Tokenizer
      eval_type: {'suffix', 'unconditional'}
                 decides whether to score a whole string or just a suffix;
                 determines the data loader for the evaluation examples.
      loss_type: {'good', 'bad'}; see utils.py
      threshold: float threshold beyond which for loss is failure
      normalize: {'token', 'example'}; average over tokens, then examples
                                       or sum over tokens and avg over ex.
    """
    self.args = args
    self.model = model
    self.tokenizer = tokenizer
    self.threshold = threshold
    self.normalize = normalize
    self.data = [json.loads(x) for x in open(args['evaluation_set'])]
    self.loss_type = loss_type
    if eval_type == 'suffix':
      self.batches = [x for x in utils.suffix_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]
    elif eval_type == 'unconditional':
      self.batches = [x for x in utils.unconditional_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]

  def evaluate(self):
    """ Runs the evaluation .

    returns average (if threshold=None) or failure rate avg
    """
    total_score = 0
    total_elts = 0
    for batch in tqdm(self.batches, desc='scoring'):
      output = self.model(batch['input_ids']).logits
      target = utils.target_of_indices(batch['input_ids'])
      scores = utils.score_suffix(output, target, batch['loss_mask'], reduction='none', reduce=False, loss_type=self.loss_type)
      if self.normalize == 'token':
        total_elts += torch.sum(batch['loss_mask']).item()
      elif self.normalize == 'example':
        total_elts += torch.sum((torch.sum(batch['loss_mask'], dim=-1)>0)).item()
      if self.threshold is not None:
        if self.normalize == 'example':
          scores = torch.sum(scores, dim=-1)
        scores = scores > self.threshold # failure rate
      total_score += torch.sum(scores).item()
    return total_score/total_elts if total_elts != 0 else total_score

class PairEvaluator:
  """
  Evaluates scores for a prefix and pair of suffixes
  under a loss function and reports the average across examples

  possibly with respect to a threshold for _failure_, thus reporting average
  failure rate.
  """

  def __init__(self, args, model, tokenizer, eval_type='suffix', diff_type='max_ratio', loss_type='balance', threshold=None, normalize='token'):
    """
    Arguments:
      args: config dictionary pertaining to the evaluation
      model: model to be evaluated (reference is stored for calling later)
      tokenizer: huggingface.Tokenizer
      eval_type: {'suffix'} only suffixes are scorable in pair mode
      loss_type: {'good-v-bad', 'balance'}; see utils.py
      threshold: float threshold beyond which for loss is failure
      normalize: {'token', 'example'}; average over tokens, then examples
                                       or sum over tokens and avg over ex.
    """
    self.args = args
    self.model = model
    self.normalize = normalize
    self.tokenizer = tokenizer
    self.data = [json.loads(x) for x in open(args['evaluation_set'])]
    self.threshold = threshold
    self.loss_type = loss_type
    if eval_type == 'suffix':
      self.batches = [x for x in utils.pair_suffix_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]

  def evaluate(self):
    total_score = 0
    total_elts = 0
    for batch in tqdm(self.batches, desc='scoring'):
      output1 = self.model(batch['input_ids1']).logits
      target1 = utils.target_of_indices(batch['input_ids1'])
      output2 = self.model(batch['input_ids2']).logits
      target2 = utils.target_of_indices(batch['input_ids2'])
      scores = utils.score_pair_suffix(output1, target1, output2, target2, batch['loss_mask1'], batch['loss_mask2'], self.loss_type, reduce=False)
      if self.normalize == 'example':
        total_elts += torch.sum((torch.sum(batch['loss_mask1'], dim=-1)>0)).item() # same for both 1 and 2
      if self.threshold is not None:
        scores = scores > self.threshold # failure rate
      scores = torch.sum(scores).item()
      total_score += scores
    return total_score/total_elts if total_elts != 0 else total_score
