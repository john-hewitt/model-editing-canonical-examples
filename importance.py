import torch
from torch import nn
import utils
from tqdm import tqdm

""" Computes sense importance scores. """
def regularized_epsilon_grad(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k=10):
  raise NotImplementedError

def regularized_epsilon_grad_per_example(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k_per_example=4):
  raise NotImplementedError

### Squared Gradient Fisher Info diag
def regularized_ewc_per_example(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k_per_example=4):
  raise NotImplementedError

def regularized_ewc(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k=10):
  raise NotImplementedError

### Attn weight stuff
def regularized_attn_weight_per_example(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k_per_example=4):
  """ Computes the per-example senses of most importance under the sum-attentions

  Sums the Backpack summation weight for each sense over all predictions of the
  canonical example (both suffixes if there are two) and subtracts out the
  average attention weight in a background corpus, sorts per example and
  takes the top-k, returns the union of the per-example top-k senses.

  Arguments:
    model: Backpack language model
    target_dataset: batches from a pair_suffix_batch_iterator or suffix_batch_iterator
    background_dataset: batches from a suffix_batch_iterator with unconditional flag
    lmbda: regularization weight for the background corpus (lambda in the paper)
    loss_helper: {'good, 'bad', 'good-v-bad', 'balance'} determines whether to use
                 attentions from the one suffix (good, bad) or both suffixes
                 (good-v-bad, balance)
    vocab_size: size of the tokenizer vocabulary
    num_senses: the number of sense vectors per type in the Backpack
    top_k_per_example: number of senses to train per example
  """
  # score containers
  target_scores = torch.zeros(vocab_size, num_senses).to(model.device)
  background_scores = torch.zeros(vocab_size, num_senses).to(model.device)
  target_topks = []

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  def get_alphas(batch, force_single=False):
    input_key = 'input_ids' if 'input_ids' in batch else 'input_ids1'
    loss_key = 'loss_mask' if 'loss_mask' in batch else 'loss_mask1'

    # Get alphas for all pairs
    if loss_type in {'good', 'bad'} or force_single:
      alphas = model(batch['input_ids']).contextualization #(bs, nv, s, s)
      bs, nv, seq, _ = alphas.shape
      # Sum alphas for source words over just target words
      loss_mask = batch['loss_mask'].reshape(bs, 1, seq, 1)
      masked_alphas = alphas*loss_mask #(bs, nv, s, s)
      alphas_for_source_words = torch.sum(masked_alphas,dim=-2) # (bs, nv, s)
      alphas_for_source_words = alphas_for_source_words.transpose(1,2).reshape(bs*seq, nv) # (bs*seq, nv)
      input_ids = batch['input_ids'].reshape(bs*seq).unsqueeze(1).expand(-1, nv) # (bs*seq, nv)
      scores_tmp = torch.zeros(vocab_size, num_senses).to(model.device) 
      scores_tmp.scatter_add_(dim=0, src=alphas_for_source_words, index=input_ids.to(torch.int64))
    elif loss_type in {'good-v-bad', 'balance'}:
      alphas1 = model(batch['input_ids1']).contextualization #(bs, nv, s, s)
      alphas2 = model(batch['input_ids2']).contextualization #(bs, nv, s, s)
      bs, nv, seq1, _ = alphas1.shape
      bs, nv, seq2, _ = alphas2.shape
      # Sum alphas for source words over just target words
      loss_mask1 = batch['loss_mask1'].reshape(bs, 1, seq1, 1)
      loss_mask2 = batch['loss_mask2'].reshape(bs, 1, seq2, 1)
      masked_alphas1 = alphas1*loss_mask1 #(bs, nv, s, s)
      masked_alphas2 = alphas2*loss_mask2 #(bs, nv, s, s)
      alphas_for_source_words1 = torch.sum(masked_alphas1,dim=-2) # (bs, nv, s)
      alphas_for_source_words2 = torch.sum(masked_alphas2,dim=-2) # (bs, nv, s)
      alphas_for_source_words1 = alphas_for_source_words1.transpose(1,2).reshape(bs*seq1, nv) # (bs*seq, nv)
      alphas_for_source_words2 = alphas_for_source_words2.transpose(1,2).reshape(bs*seq2, nv) # (bs*seq, nv)
      input_ids1 = batch['input_ids1'].reshape(bs*seq1).unsqueeze(1).expand(-1, nv) # (bs*seq, nv)
      input_ids2 = batch['input_ids2'].reshape(bs*seq2).unsqueeze(1).expand(-1, nv) # (bs*seq, nv)
      scores_tmp = torch.zeros(vocab_size, num_senses).to(model.device) 
      if loss_type == 'good-v-bad': # add good, subtract bad
        scores_tmp.scatter_add_(dim=0, src=alphas_for_source_words1, index=input_ids1.to(torch.int64))
        scores_tmp.scatter_add_(dim=0, src=-alphas_for_source_words2, index=input_ids2.to(torch.int64))
      elif loss_type == 'balance': # add both
        scores_tmp.scatter_add_(dim=0, src=alphas_for_source_words1, index=input_ids1.to(torch.int64))
        scores_tmp.scatter_add_(dim=0, src=alphas_for_source_words2, index=input_ids2.to(torch.int64))
    return scores_tmp

  # Compute losses for all background batches, aggregating gradient norms separately
  background_batch_count = 0
  for batch in tqdm(background_dataset, desc='background'):
    grads = get_alphas(batch, force_single=True)
    background_scores += (grads)
    background_batch_count += 1
  background_scores = background_scores / background_batch_count

  # Get scores for each example; normalize with background
  def get_grad(batch):
    grads = get_alphas(batch)
    normalized_grads = (grads - lmbda*background_scores)
    grads = torch.where(grads>0, normalized_grads, torch.min(normalized_grads)-1)
    sorted_scores, sorted_score_indices = torch.sort(grads.reshape(-1),descending=True)
    topk = {}
    for i in range(top_k_per_example):
      index = sorted_score_indices[i]
      topk[(index//num_senses, index%num_senses)] = sorted_scores[i]
      target_scores[index//num_senses, index%num_senses] = max(
          target_scores[index//num_senses, index%num_senses],
          sorted_scores[i],
          0.01
      )

    model.norm_backpack.epsilons.weight.grad = None
    return topk

  target_batch_count = 0
  for batch in tqdm(target_dataset, desc='target'):
    target_topks.append(get_grad(batch))

  return target_scores

def regularized_alpha(model, target_dataset, background_dataset, lmbda, loss_helper, loss_type, vocab_size=50264, num_senses=16, top_k=10):
  raise NotImplementedError
