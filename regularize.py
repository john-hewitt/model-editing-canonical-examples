import torch
from torch import nn
import utils
import json
import random
from tqdm import tqdm
import copy


class KLRegularization():
  """
  Computes a sample estimate of the KL-divergence loss
  """

  def __init__(self, orig_model, weight, config, model, tokenizer):
    self.orig_model = orig_model
    #self.original_state_dict = {c: torch.clone(original_state_dict[c]) for c in original_state_dict}
    self.weight = weight
    background_set = [json.loads(x) for x in open(config['training']['regularization_data_path'])]
    batch_size = config['training']['batch_size']
    self.device = config['device']
    self.orig_model = self.orig_model.to(self.device)
    self.batches = [x for x in utils.unconditional_batch_iterator(background_set, tokenizer, batch_size=batch_size, device='cpu')]


  def loss(self, model):
    batch = random.choice(self.batches)
    batch = {key:(value.to(self.device) if isinstance(value, torch.Tensor) else value) for key,value in batch.items()}
    orig_prediction = torch.log_softmax(self.orig_model(batch['input_ids']).logits.detach(), dim=-1)
    new_prediction = torch.log_softmax(model(batch['input_ids']).logits, dim=-1)
    loss = torch.nn.functional.kl_div(new_prediction, orig_prediction, log_target=True, reduction='batchmean')*self.weight
    return loss


class L2Regularization():
  """
  Computes weighted L2 regularization loss
  """

  def __init__(self, original_state_dict, weight):
    self.original_state_dict = {c: torch.clone(original_state_dict[c]) for c in original_state_dict}
    self.weight = weight

  def loss(self, model):
    loss = 0
    for name, parameter in model.named_parameters():
      reference_parameter = self.original_state_dict[name]
      loss += self.weight * torch.sum(torch.square(reference_parameter - parameter))
    return loss


class EWCRegularization():
  """
  Computes Elastic Weight Consolidation (Kirkpatrick et al., 2017) loss
  """

  def __init__(self, original_state_dict, weight, config, model, tokenizer):
    self.original_state_dict = {c: torch.clone(original_state_dict[c]) for c in original_state_dict}
    self.weight = weight
    background_set = [json.loads(x) for x in open(config['training']['regularization_data_path'])]
    batch_size = config['training']['batch_size']
    device = config['device']
    batches = [x for x in utils.unconditional_batch_iterator(background_set, tokenizer, batch_size=batch_size, device=device)]
    self.fisher_diag = self.estimate_fisher_diagonal(batches, model)

  def estimate_fisher_diagonal(self, batches, model):
    # Accumulate gradients
    for i, data in enumerate(batches):
      output = model(data['input_ids']).logits
      target = torch.cat((data['input_ids'][:,1:], torch.zeros(data['input_ids'].shape[0], 1).to(data['input_ids'].device)),dim=1).long()
      loss = utils.score_suffix(output, target, data['loss_mask'])
      loss.backward()
    # Take square
    fisher_diag = {}
    for name, parameter in model.named_parameters():
      fisher_diag[name] = torch.square(parameter.grad) / (i+1)
    return fisher_diag

  def loss(self, model):
    loss = 0
    for name, parameter in model.named_parameters():
      reference_parameter = self.original_state_dict[name]
      fisher_diag = self.fisher_diag[name]
      loss += self.weight * torch.sum(torch.square(fisher_diag*(reference_parameter - parameter)))
    return loss
