"""
Runs a finetuning and evaluating experiment for learning from canonical
examples. This is the main experimental entry point for the codebase.

Usage:

  python ft_experiment.py [path_to_yaml_config]

e.g.,

  python ft_experiment.py configs/example_full.yaml
"""
import yaml
import argparse
import torch
import os
import transformers
import evaluate
import regularize
import utils
import sense_finetuning
import trainer
import json

from torch.cuda.amp import autocast
import bitsandbytes as bnb


# Explicitly not performing the bnb embedding swap as it did not improve performance.
#torch.nn.Embedding(...) ->  bnb.nn.StableEmbedding(...) # recommended for NLP models
#torch.nn.Embedding = bnb.nn.StableEmbedding # recommended for NLP models


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def get_model(config):
  """
  Builds the language model to be trained, as specified by the config
  """
  if config['training']['finetune_type'] == 'senses':
    model = sense_finetuning.get_additive_sense_model(config)
  elif config['training']['finetune_type'] == 'full':
    model = utils.load_model(config['model'])
    for param in model.parameters():
      param.requires_grad = True
    if model.supports_gradient_checkpointing:
      model.gradient_checkpointing_enable()
    model = model.to(torch.bfloat16)
    model = model.to(config['device'])
  elif config['training']['finetune_type'] == 'lora':
    model = utils.load_model(config['model'])
    for param in model.parameters():
      param.requires_grad = False
    model = model.to(torch.bfloat16)
    model = model.to(config['device'])
    if model.supports_gradient_checkpointing:
      model.gradient_checkpointing_enable()
    if not isinstance(model, transformers.GPT2Model):
      model.prepare_inputs_for_generation = lambda x: x
      model.old_forward = model.forward
      model.forward = lambda *args, **kwargs: model.old_forward(input_ids=kwargs['input_ids'])

    target_modules = config['training']['lora']['target_modules']
    lora_alpha = config['training']['lora']['lora_alpha']
    r = config['training']['lora']['lora_rank']
    lora_dropout = config['training']['lora']['lora_dropout']
    peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
      target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    print(model)
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
      print(name, param.requires_grad)
  model.eval()
  return model

def get_regularization(config, model, tokenizer): 
  """
  Builds the regularization loss function for the learning process,
  as specified by the config
  """
  if 'regularization_type' not in config['training']:
    regularization_fn = lambda x: 0
  elif config['training']['regularization_type'] == 'L2':
    reg_weight = config['training']['regularization_weight']
    regularization_fn = regularize.L2Regularization(model.state_dict(), reg_weight).loss
  elif config['training']['regularization_type'] == 'EWC':
    reg_weight = config['training']['regularization_weight']
    regularization_fn = regularize.EWCRegularization(model.state_dict(), reg_weight, config, model, tokenizer).loss
  elif config['training']['regularization_type'] == 'KL':
    orig_model = get_model(config)
    orig_model.eval()
    for param in orig_model.parameters():
      param.requires_grad = False
    orig_model = orig_model.to(config['device'])
    reg_weight = config['training']['regularization_weight']
    regularization_fn = regularize.KLRegularization(orig_model, reg_weight, config, model, tokenizer).loss
  return regularization_fn


def get_intervention_eval_class(config):
  """
  Determines whether to build an evaluator over pairs or single suffixes.
  """
  if config['training']['suffix_pair']:
    return evaluate.PairEvaluator
  else:
    return evaluate.ScoreEvaluator


def exp(config, hyp=False):
  """
  Runs the finetuning experiment specified by the yaml-loaded config
  """
  device = config['device']
  if config['model'] == 'stanfordnlp/backpack-gpt2':
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model'])


  loss_type = config['training']['loss_type']

  model = get_model(config)
  threshold = config['threshold']
  normalize = config['validation']['eval_normalization']

  # Build the validation function
  degredation_targeted_path = config['validation']['degredation_targeted_path']
  degredation_general_path = config['validation']['degredation_general_path']
  intervention_eval_path = config['validation']['intervention_eval_path']
  if 'hard_negative' in config['validation']:
    hard_negative_path = config['validation']['hard_negative']['hard_negative_path']
    hard_negative_eval_type = config['validation']['hard_negative']['eval_type']
    hard_negative_eval_normalize = "token" if hard_negative_eval_type == "unconditional" else "example"
  else:
    hard_negative_path = None
  
  intervention_eval_class = get_intervention_eval_class(config)
  intervention_evaluator = intervention_eval_class(
      {'evaluation_set':intervention_eval_path}, model, tokenizer, loss_type=loss_type, threshold=threshold, normalize=normalize)
  if hard_negative_path:
    hard_negative_evaluator = evaluate.ScoreEvaluator(
        {'evaluation_set':hard_negative_path}, 
        model, tokenizer, eval_type=hard_negative_eval_type, normalize=hard_negative_eval_normalize)
  else:
    hard_negative_evaluator = None
  rest_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_targeted_path},
      model, tokenizer, eval_type='unconditional', normalize='token')
  general_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_general_path},
      model, tokenizer, eval_type='unconditional', normalize='token')
  
  def val():
    intervention_score = intervention_evaluator.evaluate()
    rest_of_prompt_score = rest_evaluator.evaluate()
    general_score = general_evaluator.evaluate()
    if hard_negative_evaluator is not None:
      hard_negative_score = hard_negative_evaluator.evaluate()
    else:
      hard_negative_score = 0
    return intervention_score, general_score, rest_of_prompt_score, hard_negative_score


  # Build the data for training
  train_set = [json.loads(x) for x in open(config['training']['dataset_path'])]
  batch_size = config['training']['batch_size']
  learning_rate = config['training']['learning_rate']
  train_batcher = utils.get_train_loader_class(config)
  batches = [x for x in train_batcher(train_set, tokenizer, batch_size=batch_size, device=device)]

  # Build the regularization function
  regularization_fn = get_regularization(config, model, tokenizer)

  # All models other than Backpacks evaluated in bfloat16 for speed
  dtype = torch.float32 if 'backpack' in config['model'] else torch.bfloat16
  print('Using dtype {}'.format(dtype))

  # Prep for posisbly saving the model
  logfile = config['logfile']
  save_info = None
  if 'save_info' in config:
    model_logdir = config['save_info']['model_logdir'] 
    if not os.path.exists(model_logdir):
      os.mkdir(model_logdir)
    save_info = config['save_info']

  # Build the right loss function for the task
  if config['training']['suffix_pair']:
    loss_helper = utils.pair_loss_batch
  else:
    loss_helper = utils.loss_batch
  if 'grad_acc_steps' in config['training']:
    grad_acc_steps = config['training']['grad_acc_steps']
  else:
    grad_acc_steps = 1

  # Train the model
  stats = trainer.train(model, batches, 0, val, learning_rate, logfile, loss_type, T_max=config['training']['num_epochs'], regularization_fn=regularization_fn, loss_helper=loss_helper, grad_acc_steps=grad_acc_steps, save_info=save_info, dtype=dtype)

  # Get the results of the last epoch underneath the loss league
  #results = utils.get_leagues(stats, hyp=hyp)
  results = {}
  results['test'] = utils.score_of_last_valid_epoch(stats, hyp=hyp)

  results['config'] = config
  with open(config['resultsfile'], 'w') as fout:
    fout.write(json.dumps(results)+'\n')
  return results, model

if __name__ == '__main__':
  argp = argparse.ArgumentParser()
  argp.add_argument('config')
  args = argp.parse_args()
  config = yaml.safe_load(open(args.config))
  if os.path.exists(config['resultsfile']):
    try:
      results = json.load(open(config['resultsfile']))
      if results['config'] == config:
        print('Results found for this experiment config. Exiting...')
        exit()
    except Exception:
      pass
  if os.path.exists(config['logfile']): # if last exp was restarted, remove logs
    print('Removing earlier log')
    os.remove(config['logfile'])
  exp(config)
