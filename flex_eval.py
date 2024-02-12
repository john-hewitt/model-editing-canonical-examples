"""
- Script will consider configs in the config dir of the format
  [f"configs/{config_dir}/{t}_{ft_type}_{league}.yaml" for t in task_names]
  - All YAMLs should contain the field save_info/model_logdir
- Merges and evaluates the resulting model
- Also evaluates each model individually to obtain baselines
  (should match results found during the finetuning run for each model, 
  but is rerun here to ensure there are no differences between machines)
"""

import yaml
import argparse
import transformers
from peft import mapping, PeftConfig, PeftModel
import re
import torch
from collections import defaultdict
import json

from torch.cuda.amp import autocast

from ft_experiment import get_intervention_eval_class
import evaluate
import utils
import norm_only_model

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

def _get_averaged_model(configs): # from https://colab.research.google.com/drive/1UmK-phTRXC4HoKb7_rScawnRqlG82svF?usp=sharing#scrollTo=bHEj3f2LkalG
  """Load models from configs and average all parameters"""
  alphal = [1 / len(configs) for _ in range(len(configs))]

  state_dict = torch.load(f"{configs[0]['save_info']['model_logdir']}/pytorch_model.bin")

  sd = {k : state_dict[k].clone() * alphal[0] for k in state_dict.keys()}
  for i, config in enumerate(configs[1:]):
    state_dict = torch.load(f"{config['save_info']['model_logdir']}/pytorch_model.bin")
    for k in state_dict.keys():
      # print(k)
      sd[k] = sd[k] + state_dict[k].clone() * alphal[i]

  model = utils.load_model(configs[0]['model'])

  print('removing unexpected keys', [x for x in sd.keys() if x not in model.state_dict().keys()])
  sd = {k: v for k, v in sd.items() if k in model.state_dict().keys()}

  model.load_state_dict(sd)
  model = model.to(configs[0]['device'])
  return model  

def _get_averaged_lora_model(configs):
  """Load peft models from configs, convert to normal models (because they may have
  different lora rank), and average all parameters from relevant layers"""

  alphal = [1 / len(configs) for _ in range(len(configs))]

  # load first model
  model = utils.load_model(configs[0]['model'])
  if not isinstance(model, transformers.GPT2Model):
    model.prepare_inputs_for_generation = lambda x: x
    model.old_forward = model.forward
    model.forward = lambda *args, **kwargs: model.old_forward(input_ids=kwargs['input_ids'])
  print("configs[0]['save_info']['model_logdir']", configs[0]['save_info']['model_logdir'])
  model = PeftModel.from_pretrained(model, configs[0]['save_info']['model_logdir'])

  peft_state_dict = model.state_dict()

  # load a list of layers to consider
  layers_to_consider = set()
  for model_logdir in [config['save_info']['model_logdir'] for config in configs]:
    cur_peft_config = mapping.PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(model_logdir)].from_pretrained(model_logdir)
    regex_pattern = cur_peft_config.target_modules+"\.(weight|bias)"
    for k in peft_state_dict.keys():
      if re.fullmatch(regex_pattern, k) is not None:
        layers_to_consider.add(k)
  layers_to_consider = [x.split('base_model.model.')[1] for x in layers_to_consider]

  model = model.merge_and_unload()
  state_dict = model.state_dict()

  # get averaged state dict
  sd = {}
  for k in state_dict.keys():
    if k in layers_to_consider: # only get average for lora-affected layers
      print("Averaging", k)
      sd[k] = state_dict[k].clone() * alphal[0]
    else: 
      sd[k] = state_dict[k].clone()

  for i, config in enumerate(configs[1:]):
    cur_model = utils.load_model(config['model'])
    if not isinstance(cur_model, transformers.GPT2Model):
      cur_model.prepare_inputs_for_generation = lambda x: x
      cur_model.old_forward = cur_model.forward
      cur_model.forward = lambda *args, **kwargs: cur_model.old_forward(input_ids=kwargs['input_ids'])
    cur_model = PeftModel.from_pretrained(cur_model, config['save_info']['model_logdir'])
    cur_model = cur_model.merge_and_unload()

    state_dict = cur_model.state_dict()
    for k in layers_to_consider:
      sd[k] = sd[k] + state_dict[k].clone() * alphal[i]

  model.load_state_dict(sd)
  model = model.to(configs[0]['device'])

  model.forward = model.old_forward 
  return model, layers_to_consider

def _merge_senses(model, sense_change_weights):
  n_embd = model.norm_backpack.backpack.config.n_embd
  merged_senses = torch.zeros_like(model.norm_backpack.sense_change_vecs.weight)

  # sum senses
  token_sense_counts = {}
  for sense_dict in sense_change_weights:
    for token_id in sense_dict:
      for target_sense in sense_dict[token_id]:
        if token_id not in token_sense_counts:
          token_sense_counts[token_id] = defaultdict(int)
        token_sense_counts[token_id][target_sense] += 1
        merged_senses[token_id][n_embd * target_sense : n_embd * (target_sense+1)] += sense_dict[token_id][target_sense]

  # get average 
  for token_id in token_sense_counts:
    for target_sense in token_sense_counts[token_id]:
      if token_sense_counts[token_id][target_sense] > 1:
        # in the event of multiple changes
        print(f"MERGING Token {token_id} ({tokenizer.decode([token_id])}) sense {target_sense} has {token_sense_counts[token_id][target_sense]} counts")
      merged_senses[token_id][n_embd * target_sense : n_embd * (target_sense+1)] /= token_sense_counts[token_id][target_sense]
  return merged_senses, token_sense_counts

def load_model_union(configs, return_info=False, used_save_mode='best'):
  """Load a union of models"""

  edited_info = None
  
  all_config_types = [config['training']['finetune_type'] for config in configs]
  assert len(set(all_config_types)) == 1, "All configs must have the same finetune type"

  finetune_type = all_config_types[0]
  device = configs[0]['device']
  assert 'save_info' in configs[0].keys()
  if finetune_type == 'senses':
    # get a base model
    model = utils.load_model(configs[0]['model'])
    model = norm_only_model.NormBackpackLM(model, train_senses_low=True, senses_to_change={})
    model = model.to(device)
    model.device = device

    # merge and freeze the senses
    if used_save_mode == 'best':
      sense_change_weights = [torch.load(f"{config['save_info']['model_logdir']}/best.pt") for config in configs]
    else:
      sense_change_weights = [torch.load(f"{config['save_info']['model_logdir']}/epoch{config['training']['num_epochs']-1}.pt") for config in configs]
    model.norm_backpack.sense_change_vecs.weight.data, token_sense_counts = _merge_senses(model, sense_change_weights)
    model.norm_backpack.sense_change_vecs.weight.requires_grad = False
    # update sense_change_selector as well
    for voc_index in token_sense_counts:
      for sense_index in token_sense_counts[voc_index]:
        model.norm_backpack.sense_change_selector.weight.data[voc_index, sense_index] = 1

  elif finetune_type == 'full':
    model = _get_averaged_model(configs)

  elif finetune_type == 'lora':
    model, edited_info = _get_averaged_lora_model(configs)

  else:
    raise NotImplementedError(f"Finetune type {finetune_type} not implemented")

  model = model.to(device)
  # model.device = device
  model.eval()
  for _, param in model.named_parameters():
    param.requires_grad = False 

  model = model.to(torch.bfloat16)

  if not return_info:
    return model 
  else:
    return model, edited_info

def eval_model_on_config(model, config, cached_general_score):
  loss_type = config['training']['loss_type']

  # Build the validation function
  degredation_targeted_path = config['validation']['degredation_targeted_path']
  degredation_general_path = config['validation']['degredation_general_path']
  intervention_eval_path = config['validation']['intervention_eval_path']

  threshold = config['threshold']
  normalize = config['validation']['eval_normalization']

  intervention_eval_class = get_intervention_eval_class(config)
  intervention_evaluator = intervention_eval_class(
      {'evaluation_set':intervention_eval_path}, model, tokenizer, loss_type=loss_type, threshold=threshold, normalize=normalize)
  rest_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_targeted_path},
      model, tokenizer, eval_type='unconditional', normalize='token')
  general_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_general_path},
      model, tokenizer, eval_type='unconditional', normalize='token')
  
  model.eval()
  intervention_score = intervention_evaluator.evaluate()
  rest_of_prompt_score = rest_evaluator.evaluate()
  if len(cached_general_score) > 0:
    assert degredation_general_path == cached_general_score['degredation_general_path']
    general_score = cached_general_score['general_score']
  else:
    general_score = general_evaluator.evaluate()

  # cache results
  cached_general_score['degredation_general_path'] = degredation_general_path
  cached_general_score['general_score'] = general_score

  return {
    'intervention_score': intervention_score,
    'general_score': general_score,
    'rest_of_prompt_score': rest_of_prompt_score,
  }


if __name__ == '__main__':
  argp = argparse.ArgumentParser()
  argp.add_argument('league', type=float)
  argp.add_argument('ft_type')
  argp.add_argument('config_dir') # dir of yaml files of the expected format
  argp.add_argument('logfile') # file to write evaluation results to

  args = argp.parse_args()
  league = args.league
  ft_type = args.ft_type
  config_dir = args.config_dir
  logfile = args.logfile
  assert ft_type in ['full', 'senses', 'lora']
  assert league in [0.001, 1e-4, 1e-5]
  print("logfile", args.logfile)

  task_names = [
    'company', 
    'country', 
    'gender', 
    'stereoset', 
    'temporal',
    'verb',
  ]

  config_files = [f"configs/{config_dir}/{t}_{ft_type}_{league}.yaml" for t in task_names]
  print("config_files", config_files)
  configs = [yaml.safe_load(open(filename)) for filename in config_files]
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

  merged_model = load_model_union(configs)

  # evaluate
  merged_results = {}

  cached_general_score = {} 
  for i, config in enumerate(configs):
    with autocast(dtype=torch.bfloat16):
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        merged_results[task_names[i]] = eval_model_on_config(merged_model, config, cached_general_score)

  # get baselines
  unmerged_results = {}
  for i, config in enumerate(configs):
    if ft_type == 'senses':
      cur_model = load_model_union([config])
      
    elif ft_type == 'lora':
      cur_model = utils.load_model(config['model'])
      if not isinstance(cur_model, transformers.GPT2Model):
        cur_model.prepare_inputs_for_generation = lambda x: x
        cur_model.old_forward = cur_model.forward
        cur_model.forward = lambda *args, **kwargs: cur_model.old_forward(input_ids=kwargs['input_ids'])
      cur_model = PeftModel.from_pretrained(cur_model, config['save_info']['model_logdir'])
      # cur_model = cur_model.merge_and_unload() # TODO: figure this out
      
    else:
      cur_config = transformers.AutoConfig.from_pretrained(config['model'], trust_remote_code=True)
      cur_model = transformers.AutoModelForCausalLM.from_pretrained(
        config['save_info']['model_logdir'], config=cur_config, trust_remote_code=True,
      )

    cur_model = cur_model.to(torch.bfloat16)
    cur_model = cur_model.cuda()
    cur_model.eval()

    cached_general_score = {} 
    with autocast(dtype=torch.bfloat16):
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        unmerged_results[task_names[i]] = eval_model_on_config(cur_model, config, cached_general_score)

  all_results = {
    'merged': merged_results,
    'unmerged': unmerged_results,
  }
  with open(logfile, 'w') as fh:
    print(json.dumps(all_results), file=fh)
