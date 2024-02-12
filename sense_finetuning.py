import utils
import json
import importance
import norm_only_model
import transformers

def get_sense_scores(model, tokenizer, train_set, background_set, lmbda, loss_helper, loss_type, top_k_per_example=0, top_k=0, device='cuda', train_batcher=utils.suffix_batch_iterator, sense_method=None):
  """
  Figures out which senses to finetune depending on the importance method chosen
  """

  # Determine important senses
  model.eval()
  batch_size = 1
  batches = [x for x in train_batcher(train_set, tokenizer, batch_size=batch_size, device=device)]
  background_batches = [x for x in utils.unconditional_batch_iterator(background_set, tokenizer, batch_size=batch_size, device=device)]
  if top_k_per_example != 0 and top_k != 0:
    raise ValueError("Sense maxes must be either per example or overall, not both")
  elif top_k_per_example > 0:
    if sense_method == 'fisher':
      importance_scores = importance.regularized_ewc_per_example(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k_per_example=top_k_per_example)
    elif sense_method == 'epsilon_grad':
      importance_scores = importance.regularized_epsilon_grad_per_example(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k_per_example=top_k_per_example)
    elif sense_method == 'alpha':
      importance_scores = importance.regularized_attn_weight_per_example(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k_per_example=top_k_per_example)
    else:
      raise ValueError('Unknown sense method: {}'.format(sense_method))
  elif top_k > 0:
    if sense_method == 'fisher':
      importance_scores = importance.regularized_ewc(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k=top_k)
    elif sense_method == 'epsilon_grad':
      importance_scores = importance.regularized_epsilon_grad(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k=top_k)
    elif sense_method == 'alpha':
      importance_scores = importance.regularized_alpha(model, batches, background_batches, lmbda=lmbda, loss_helper=loss_helper, loss_type=loss_type, top_k=top_k)
    else:
      raise ValueError('Unknown sense method: {}'.format(sense_method))
  all_scores = {}
  scores_to_change = []
  for voc in range(50257):
    for ell in range(16):
      all_scores[(voc, ell)] = importance_scores[voc,ell]
  for i in sorted(all_scores, key=lambda x: -all_scores[x]):
    if all_scores[i] > 0:
      print(tokenizer.decode(i[0]), i[1], all_scores[i])
      scores_to_change.append((i[0], i[1]))
    else:
      break
  return scores_to_change

def get_additive_sense_model(config, device='cuda'):
  """
  Builds the sense finetuning Backpack with the chosen senses
  """

  # Loss specs
  if config['training']['suffix_pair']:
    loss_helper = utils.pair_loss_batch
  else:
    loss_helper = utils.loss_batch
  loss_type = config['training']['loss_type']

  train_batcher = utils.get_train_loader_class(config)

  regularization_lambda = config['senses']['regularization_lambda']

  # Get scores
  model = utils.load_model(config['model'])
  train_set = [json.loads(x) for x in open(config['training']['dataset_path'])]
  background_set = [json.loads(x) for x in open(config['senses']['background_data_path'])]
  model = norm_only_model.NormBackpackLM(model, train_senses_low=True, senses_to_change='all')
  model.eval()
  model = model.to(device)
  model.device = device
  sense_method = 'epsilon_grad' if 'sense_method' not in config['senses'] else config['senses']['sense_method']
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  scores_to_change = get_sense_scores(model, tokenizer,
      train_set,
      background_set,
      lmbda=regularization_lambda,
      loss_helper=loss_helper,
      loss_type=loss_type,
      top_k_per_example=config['senses']['max_senses_per_example'],
      top_k=config['senses']['max_senses_total'],
      train_batcher=train_batcher,
      sense_method=sense_method
      )

  # Make the model
  model = utils.load_model(config['model'])
  model = norm_only_model.NormBackpackLM(model, train_senses_low=True, senses_to_change=scores_to_change)
  model.eval()
  for param in model.parameters():
    param.requires_grad = False
  model.norm_backpack.sense_change_vecs.weight.requires_grad = True
  model = model.to(device)
  model.device = device
  return model
