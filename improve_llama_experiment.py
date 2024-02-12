"""
Runs an experiment ensembing a finetune-pretrain backpack ratio
with GPT-J-6B.
"""

import yaml
import argparse
import torch
from transformers import AutoModelForCausalLM
from torch import nn
import norm_only_model
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
import ft_experiment

def update_backpack(backpack, sense_dict):
  """
  Load in senses for a Backpack
  """
  n_embd = backpack.norm_backpack.backpack.config.n_embd
  sense_tensor = torch.zeros_like(backpack.norm_backpack.sense_change_vecs.weight).to(backpack.norm_backpack.sense_change_vecs.weight.device)
  for token_id in sense_dict:
    for target_sense in sense_dict[token_id]:
      sense_tensor[token_id][n_embd * target_sense : n_embd * (target_sense+1)] += sense_dict[token_id][target_sense].to(sense_tensor.device)
  backpack.norm_backpack.sense_change_vecs.weight = nn.Parameter(sense_tensor)

if __name__ == '__main__':
  argp = argparse.ArgumentParser()
  argp.add_argument('config')
  args = argp.parse_args()
  config = yaml.safe_load(open(args.config))

  if os.path.exists(config['logfile'] + '.intervene'):
    print('Result found. Exiting...')
    exit()
    
  # Get Backpack models
  backpack = ft_experiment.get_model(config)
  update_backpack(backpack, torch.load(config['save_info']['model_logdir'] + '/best.pt'))
  for param in backpack.parameters():
    param.requires_grad = False
  backpack.eval()

  # Get LLAMA model
  model_id = 'EleutherAI/gpt-j-6b'
  #model_id = 'gpt2'
  llama_config = transformers.AutoConfig.from_pretrained(model_id)
  large_model = AutoModelForCausalLM.from_pretrained(model_id, config=llama_config, cache_dir='/juice4/scr4/nlp/backpacks/transformer')
  for param in large_model.parameters():
    param.requires_grad = False
  large_model = large_model.to(torch.bfloat16)
  large_model = large_model.to(config['device'])
  large_model.eval()

  # Get combined model
  model = norm_only_model.LLAMAWithABackpack(backpack, large_model, weight=1)
  model.eval()
  for param in model.parameters():
    param.requires_grad = False
  #print(combined_model(torch.zeros(1,512).to('cuda').long()))

  # Evaluate
  device = config['device']
  model.device = device
  if config['model'] == 'stanfordnlp/backpack-gpt2':
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model'])

  loss_type = config['training']['loss_type']

  threshold = config['threshold']
  normalize = config['validation']['eval_normalization']

  # Build the validation function
  degredation_targeted_path = config['validation']['degredation_targeted_path']
  degredation_general_path = config['validation']['degredation_general_path']
  intervention_eval_path = config['validation']['intervention_eval_path']
  hard_negative_path = config['validation']['hard_negative']['hard_negative_path']
  hard_negative_eval_type = config['validation']['hard_negative']['eval_type']
  hard_negative_eval_normalize = "token" if hard_negative_eval_type == "unconditional" else "example"
  
  intervention_eval_class = ft_experiment.get_intervention_eval_class(config)
  intervention_evaluator = intervention_eval_class(
      {'evaluation_set':intervention_eval_path}, model, tokenizer, loss_type=loss_type, threshold=threshold, normalize=normalize)
  hard_negative_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':hard_negative_path}, 
      model, tokenizer, eval_type=hard_negative_eval_type, normalize=hard_negative_eval_normalize)
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
    hard_negative_score = hard_negative_evaluator.evaluate()
    return intervention_score, general_score, rest_of_prompt_score, hard_negative_score

  def just_general_val():
    general_score = general_evaluator.evaluate()
    return general_score

  model.weight = 0
  score, gpt_j_loss, rest, hard = val()
  initial_dict = {'intervention': score, 'general': gpt_j_loss, 'hard_negative': hard, 'rest': rest}
  print('Original loss:', gpt_j_loss)
  
  # Approximate the highest weight we can put on the Backpack
  guess = 1.1
  league = float(config['league'])
  while True:
    guess = guess - 0.1
    print('Guess: {}'.format(guess))
    model.weight = guess
    general = just_general_val()
    if general < (1+league)*gpt_j_loss:
      break

  # Evaluate and report
  score, gen, rest, hard = val()
  with open(config['logfile'] + '.intervene', 'w') as fout:
    fout.write(json.dumps({'weight': guess, 'intervention': score, 'general': gen, 'hard_negative': hard, 'rest': rest, 'config': config, 'initial': initial_dict}) +'\n')
