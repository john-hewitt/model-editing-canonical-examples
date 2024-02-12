"""
Modified Backpack Language Models for sense finetuning
and ensembling with a larger model
"""
import torch
from torch import nn

from torch.cuda.amp import autocast

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

@dataclass
class BackpackGPT2BaseModelOutput(ModelOutput):
    """
    modification: tensor of learned sense differences for output
    """
    hidden_states: torch.FloatTensor = None
    contextualization: torch.FloatTensor = None
    senses: torch.FloatTensor = None
    modification: torch.FloatTensor = None

class NormBackpack(nn.Module):
  """
  Backpack that allows one to finetune individual senses (word, sense_index)
  in the model_dim space.
  """
  def __init__(self, backpack, train_epsilon=False, train_senses_low=False, train_senses_vocab=False, senses_to_change=set()):
    """

    Arguments:
      backpack: a loaded Backpack LM
      train_epsilon: should be False; this is vestigial
      train_senses_low: whether to train senses in the low (model) dimensionality
      train_senses_vocab: should be False; this is vestigial
      senses_to_change: a set of (vocab_index, sense_index) pairs specifying which
                        senses can be updated
    """
    super().__init__()
    self.backpack = backpack
    self.train_epsilon = train_epsilon
    self.train_senses_low = train_senses_low
    self.train_senses_vocab = train_senses_vocab
    self.num_senses = backpack.config.num_senses
    self.epsilons = nn.Embedding(backpack.config.vocab_size, backpack.config.num_senses)
    self.epsilons.weight = torch.nn.Parameter(torch.ones(backpack.config.vocab_size, backpack.config.num_senses))
    self.senses_to_change = senses_to_change
    
    if self.train_senses_low:
      # Make the selector of where to use the new senses
      self.sense_change_selector = nn.Embedding(backpack.config.vocab_size, backpack.config.num_senses)
      self.sense_change_selector.weight = nn.Parameter(torch.zeros(backpack.config.vocab_size, backpack.config.num_senses))
      if isinstance(senses_to_change, list):
        for voc_index, sense_index in senses_to_change:
          self.sense_change_selector.weight.data[voc_index, sense_index] = 1
      elif senses_to_change == 'all':
        self.sense_change_selector.weight.data = torch.ones_like(self.sense_change_selector.weight.data).to(self.sense_change_selector.weight.data.device)
      else:
        raise ValueError("Wrong type for senses_to_change")

      # Make the sense change parameters
      self.sense_change_vecs = nn.Embedding(backpack.config.vocab_size, backpack.config.num_senses*backpack.config.n_embd)
      self.sense_change_vecs.weight = nn.Parameter(torch.zeros(backpack.config.vocab_size, backpack.config.num_senses*backpack.config.n_embd))

  def forward(self, input_ids, position_ids=None, apply_modification=True, contextualization=None, senses=None):
      """
      Arguments:
        input_ids: torch.tensor of longs tokenized input, shape (bs, seqlen)
        position_ids: assumed to be None
        apply_modification: whether to apply the learned sense updates
        contextualization: optional pre-computed contextualization tensor
        senses: optional pre-computed base senses

      Returns:
        BackpackGPT2BaseModelOutput
      """
      bs, seqlen = input_ids.shape
      if not self.train_epsilon:
        self.epsilons.requires_grad = False
      
      # Make senses
      if senses is None:
        senses = self.backpack.word_embeddings(input_ids)
        senses = self.backpack.sense_network(senses) # (bs, nv, s, d)

      if self.train_senses_low and apply_modification:
        selection = self.sense_change_selector(input_ids).transpose(1,2) #(bs, nv, s)
        updates = self.sense_change_vecs(input_ids).reshape(bs, seqlen, self.num_senses, -1).transpose(1,2) #(bs, nv, s, d)
        modification = torch.where(selection.bool().unsqueeze(3).expand(-1,-1,-1,updates.shape[-1]), updates, torch.zeros_like(updates).to(selection.device))
        senses += modification
      else:
        modification = None

      # Weight senses
      epsilons = self.epsilons(input_ids) #(bs, s, nv)
      senses = senses* epsilons.unsqueeze(3).transpose(1,2) # (bs, nv, s, d)

      if contextualization is None:
        contextl_hidden_states = self.backpack.gpt2_model(input_ids, position_ids=position_ids).last_hidden_state # (bs, s, d)
        contextualization = self.backpack.sense_weight_net(contextl_hidden_states) # (bs, nv, s, s)

      # Compute resulting outputs
      hidden_states = torch.sum(contextualization @ senses, dim=1) # (bs, nv, s, d) -> (bs, s, d)
      return BackpackGPT2BaseModelOutput(
          hidden_states=hidden_states,
          contextualization=contextualization,
          senses=senses,
          modification=modification
      )

@dataclass
class BackpackGPT2LMHeadModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    contextualization: torch.FloatTensor = None
    sense_diffs: torch.FloatTensor = None
    senses: torch.FloatTensor = None
    backpack_modification: torch.FloatTensor = None

class NormBackpackLM(nn.Module):

  def __init__(self, backpack_lm, **kwargs):
    super().__init__()
    self.norm_backpack = NormBackpack(backpack_lm.backpack, **kwargs)
    self.lm_head = backpack_lm.lm_head

  def forward(self, input_ids, position_ids=None):
      outputs = self.norm_backpack(input_ids, position_ids=position_ids)
      hidden_states, contextualization = outputs.hidden_states, outputs.contextualization
      lm_logits = self.lm_head(hidden_states) # (bs, s, V)
      return BackpackGPT2LMHeadModelOutput(
            logits=lm_logits,
            contextualization=contextualization,
        )

class LLAMAWithABackpack(nn.Module):
  """
  Combines the logits of a Backpack difference (finetuned - pretrained)
  with another LM.
  """

  def __init__(self, backpack_lm, llama_lm, weight=1,**kwargs):
    super().__init__()
    self.norm_backpack = backpack_lm.norm_backpack
    self.lm_head = backpack_lm.lm_head
    self.llama_lm = llama_lm
    self.weight = weight

  def forward(self, input_ids, position_ids=None, backpack_weight=None, return_components=False):
      
      # Without modification
      outputs = self.norm_backpack(input_ids, position_ids=position_ids, apply_modification=False)
      hidden_states, contextualization, senses = outputs.hidden_states, outputs.contextualization, outputs.senses
      untuned_backpack_lm_logits = torch.log_softmax(self.lm_head(hidden_states), dim=-1) # (bs, s, V)

      if backpack_weight is None:
        backpack_weight = self.weight

      # With modification
      outputs = self.norm_backpack(input_ids, position_ids=position_ids, contextualization=contextualization, senses=senses)
      hidden_states, contextualization, modification = outputs.hidden_states, outputs.contextualization, outputs.modification
      tuned_backpack_lm_logits = torch.log_softmax(self.lm_head(hidden_states), dim=-1) # (bs, s, V)

      backpack_modification = tuned_backpack_lm_logits - untuned_backpack_lm_logits

      # LLAMA
      with autocast(dtype=torch.bfloat16):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
          llama_logits = self.llama_lm(input_ids).logits

      # Combination
      lm_logits = llama_logits[:,:,:50257] + backpack_weight*backpack_modification[:,:,:50257]

      return BackpackGPT2LMHeadModelOutput(
            logits=lm_logits,
            backpack_modification=backpack_modification,
            sense_diffs=modification,
            senses=senses,
            contextualization=contextualization
        )
