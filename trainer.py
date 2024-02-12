"""
Finetuning a model on a dataset
"""

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM
from torch.cuda.amp import autocast
import argparse
import json
import utils
from utils import loss_batch
import evaluate
import datasets
import random
import sys
import norm_only_model
import importance

import bitsandbytes as bnb

from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = 'cuda'


def train(model, batches, flip, val_fn, lr, val_file, loss_type, T_max=50, regularization_fn=None, loss_helper=loss_batch, grad_acc_steps=1, save_info=None, dtype=torch.float16):


  optimizer = bnb.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.995), optim_bits=8)
  scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

  ce = nn.CrossEntropyLoss(reduction='none')

  model.eval()
  model = model.to(dtype)

  # stats
  stats = []

  # Training loop
  if T_max != -1:
    with autocast(dtype=dtype):
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        cap, gen, rest, hard_neg = val_fn()
        with open(val_file, 'a') as fout:
          if T_max == 0: # If just scoring, we can score bigger models
            for parameter in model.parameters():
              parameter.requires_grad = False
          loss_mean = 0 
          for i, data in enumerate(batches):
            loss = loss_helper(model, data, loss_type)
            loss_mean += loss.item()
          stats.append({'intervention': cap, 'general': gen, 'train_loss': loss_mean/(i+1), 'rest_of_prompt':rest, 'hard_negative': hard_neg})
          fout.write(json.dumps(stats[-1]) + '\n')
  
  best_intervention_val = None
  for epoch in range(T_max):
    random.shuffle(batches)
    loss_mean = 0 
    grad_acc_intermediate = 0
    for i, data in enumerate(batches):
      optimizer.zero_grad()
      with autocast(dtype=dtype):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
          loss = loss_helper(model, data, loss_type)
          loss.backward()
          grad_acc_intermediate += 1
          reg_loss = regularization_fn(model)
          if isinstance(reg_loss, torch.Tensor):
            reg_loss.backward()
          if grad_acc_intermediate == grad_acc_steps:
            optimizer.step()
            grad_acc_intermediate = 0
          loss_mean += loss.item()
    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss_mean/(i+1)}, LR: {scheduler.get_last_lr()[0]}")
    scheduler.step()
    with autocast(dtype=dtype):
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        cap, gen, rest, hard_neg = val_fn()
    with open(val_file, 'a') as fout:
      stats.append({'intervention': cap, 'general': gen, 'train_loss': loss_mean/(i+1), 'rest_of_prompt':rest, 'hard_negative': hard_neg})
      fout.write(json.dumps(stats[-1]) + '\n')


    if save_info is not None:
      model_logdir = save_info['model_logdir']

      # save by epoch
      if save_info['criteria'] == 'epoch':
        if hasattr(model, "norm_backpack"): # do not save senses for full finetuning
          if epoch % 10 == 0 or epoch == T_max - 1:
            save_model_senses(model, f"{model_logdir}/epoch{epoch}.pt")
        else: # save weights from full finetuning
          if epoch == T_max - 1:
            model.save_pretrained(model_logdir)

      # save by last valid performance in indicated league
      elif save_info['criteria'] == 'league':
        league = float(save_info['league'])
        league_loss_cutoff = stats[0]['general'] * (1+league)
        
        if gen < league_loss_cutoff: # don't check if best so far
          best_intervention_val = cap 
          if hasattr(model, "norm_backpack"): 
            save_model_senses(model, f"{model_logdir}/best.pt")
          else: # save weights from full finetuning
            model.save_pretrained(model_logdir)
      else:
        raise ValueError

    loss_mean = 0
  return stats

def save_model_senses(model, outpath, n_embd=768):
  save_obj = {}
  for token_id, target_sense in model.norm_backpack.senses_to_change:
    if token_id not in save_obj:
      save_obj[token_id] = {}
    save_obj[token_id][target_sense] = model.norm_backpack.sense_change_vecs.weight[token_id][n_embd * target_sense : n_embd * (target_sense+1)].detach().clone().half()
  torch.save(save_obj, outpath)
