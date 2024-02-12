import yaml
import random
import numpy as np
import math
import re

task = 'gender'

TASK_INFO = {
    'company': {'eval_type': 'suffix', 'path': 'company_ceo_hard_neg_eval_clear'},
    'country': {'eval_type': 'suffix', 'path': 'country_capital_hard_neg'},
    'gender': {'eval_type': 'suffix', 'path': 'pronoun_gender_bias_hard_neg_eval'},
    'temporal': {'eval_type': 'suffix', 'path': 'temporal_hard_neg_eval_clear'},
    'verb': {'eval_type': 'suffix', 'path': 'verb_conjugation_hard_neg_eval'},
    'stereoset': {'eval_type': 'suffix', 'path': 'stereoset_hard_neg'},
}


#threshold = -np.log(0.05).item()
threshold = np.abs(-np.log(.1/1.5)+np.log(.1)).item() #.405

np.random.seed(876545987)
random.seed(876545987)

# full learning rate
lr_distribution = lambda: np.power(10.0, -np.random.uniform(4,8.5)).item()
# lora learning rate
lora_lr_distribution = lambda: np.power(10.0, -np.random.uniform(2,6.5)).item()
# sense learning rate
#sense_lr_distribution = lambda: np.power(10.0, -np.random.uniform(2.5,4.5)).item()
#sense_lr_distribution = lambda: np.power(10.0, -np.random.uniform(2,4)).item()
sense_lr_distribution = lambda: np.power(10.0, -np.random.uniform(1.5,4)).item()

# batch size
batch_size = 10

reg_distribution = lambda: np.power(10.0, np.random.uniform(-1,0)).item()
#reg_distribution = lambda: 0

# LoRA 
lora_start_index_args = {'a': 0, 'b': 11}
start_index_distribution = lambda: random.randint(**lora_start_index_args)
def lora_distribution():
  start_index = start_index_distribution()
  lora_length_args = {'a': 0, 'b': 11-start_index}
  length = random.randint(**lora_length_args)
  layer_string = '|'.join([str(x) for x in list(range(start_index, start_index+length+1))])
  return '.*\.({})\.mlp\.(c_proj|c_fc)'.format(layer_string)

lora_layers_percent_distribution = lambda: np.random.uniform(.1, .9)

layer_count_dictionary = {
    #'stanfordnlp/backpack-gpt2': 12,
    #'meta-llama/Llama-2-7b-hf': 36
    'EleutherAI/gpt-j-6b': 28,
    #'EleutherAI/pythia-70m': 6,
    #'EleutherAI/pythia-160m': 12,
    #'EleutherAI/pythia-410m': 24,
    #'EleutherAI/pythia-1b': 16,
    #'EleutherAI/pythia-1.4b': 24,
    #'EleutherAI/pythia-2.8b': 32,
    #'EleutherAI/pythia-6.9b': 32,
    #'meta-llama/Llama-2-7b-hf': 32,
}

# Sense stuff
#sense_count_distribution = lambda: int(np.power(10.0, np.random.uniform(.5, 1.5)))
sense_count_distribution = lambda: np.random.randint(1, 10)
sense_count_distribution = lambda: np.random.randint(5, 12)
#sense_count_distribution = lambda: np.random.randint(1, 5)
sense_regularization_distribution = lambda: int(np.power(10.0, np.random.uniform(3,10)))
sense_regularization_distribution = lambda: 1000
#sense_regularization_distribution = lambda: int(np.power(10.0, np.random.uniform(2,3.5)))

for seed in (0,1):
  for model in layer_count_dictionary:
    safe_model = model.replace('/', '-')

    lora_rank_args = {'a': 1, 'b': 256}
    lora_rank_distribution = lambda: random.randint(**lora_rank_args)

    # regularization_type
    #regtype_distribution = lambda: random.sample(('L2', 'EWC', 'KL'), k=1)[0]
    regtype_distribution = lambda: random.sample(('KL',), k=1)[0]

    # Sense regularization
    #sense_regularization_args = {'sigma': 4, 'mean':15}
    #sense_regularization_distribution = lambda: np.random.lognormal(**sense_regularization_args)

    # Epoch count
    #epoch_count_args = {'a':5, 'b': 25}
    #epoch_count_distribution = lambda: random.randint(**epoch_count_args)

    epoch_count = 10

    # Total rounds
    SWEEP_SIZE = 25

    # full finetuning
    for i in range(SWEEP_SIZE):
      lr = lr_distribution()
      batch_size = batch_size
      regularization_type = regtype_distribution()
      regularization_weight = reg_distribution()
      #epoch_count = epoch_count_distribution()

      config = yaml.safe_load(open('example_full.yaml'))
      config['threshold'] = threshold
      config['model'] = model
      config['seed'] = seed
      config['training']['grad_acc_steps'] = 5
      config['training']['batch_size'] = 1
      config['training']['num_epochs'] = epoch_count
      config['training']['learning_rate'] = lr
      config['training']['regularization_type'] = regularization_type
      config['training']['regularization_weight'] = regularization_weight
      config['training']['regularization_data_path'] = 'data/trainval-chunked.jsonl'
      # Hard negs
      eval_path = config['validation']['intervention_eval_path']
      config['validation']['hard_negative'] = {}
      config['validation']['hard_negative']['hard_negative_path'] = eval_path[:eval_path.index('split/') + 6] + TASK_INFO[task]['path'] + "-val.jsonl"
      config['validation']['hard_negative']['eval_type'] = TASK_INFO[task]['eval_type']
      config['logfile'] = 'gptjresults/{}-gender-full-lr{:.2E}-epochs{}-regtype{}-regw{:.2E}.out.seed{}'.format(
          safe_model,
          lr,
          epoch_count,
          regularization_type,
          regularization_weight,
          seed
          )
      config['resultsfile'] = 'gptjresults/{}-gender-full-lr{:.2E}-epochs{}-regtype{}-regw{:.2E}.results.out.seed{}'.format(
          safe_model,
          lr,
          epoch_count,
          regularization_type,
          regularization_weight,
          seed
          )
      with open('{}-full.{}.sweep.yaml'.format(safe_model, i), 'w') as fout:
        yaml.dump(config, fout)

    # LoRA finetuning
    for i in range(SWEEP_SIZE):
      lr = lora_lr_distribution()
      batch_size = batch_size
      regularization_type = regtype_distribution()
      regularization_weight = reg_distribution()

      lora_rank = lora_rank_distribution()
      lora_layer_middle = layer_count_dictionary[model]//2
      lora_percent = lora_layers_percent_distribution()
      lora_layers_count = math.ceil(layer_count_dictionary[model]*lora_percent)
      if lora_layers_count % 2 == 0:
        lora_layer_start = lora_layer_middle - lora_layers_count//2
        lora_layer_end = lora_layer_middle + lora_layers_count//2
      else:
        lora_layer_start = lora_layer_middle - lora_layers_count//2
        lora_layer_end = lora_layer_middle + lora_layers_count//2+1
      layer_string = '|'.join([str(x) for x in list(range(lora_layer_start, lora_layer_end+1))])
      lora_target_string = '.*\.({})\.mlp\.(fc_in|fc_out)'.format(layer_string)
      sanitized_target = re.search(r'\((([0-9][0-9]?\|?)+)', lora_target_string).group()[1:].split('|')
      sanitized_target = sanitized_target[0] + 'to' + sanitized_target[-1]

      config = yaml.safe_load(open('example_lora.yaml'))
      config['threshold'] = threshold
      config['model'] = model
      config['seed'] = seed
      config['training']['grad_acc_steps'] = 5
      config['training']['batch_size'] = 1
      config['training']['num_epochs'] = epoch_count
      config['training']['learning_rate'] = lr
      config['training']['regularization_type'] = regularization_type
      config['training']['regularization_weight'] = regularization_weight
      config['training']['regularization_data_path'] = 'data/trainval-chunked.jsonl'
      # Hard negs
      eval_path = config['validation']['intervention_eval_path']
      config['validation']['hard_negative'] = {}
      config['validation']['hard_negative']['hard_negative_path'] = eval_path[:eval_path.index('split/') + 6] + TASK_INFO[task]['path'] + "-val.jsonl"
      config['validation']['hard_negative']['eval_type'] = TASK_INFO[task]['eval_type']
      config['training']['lora']['target_modules'] = lora_target_string
      config['training']['lora']['lora_rank'] = lora_rank
      config['logfile'] = 'gptjresults/{}-gender-lora-lr{:.2E}-epochs{}-regtype{}-regw{:.2E}-lorarank{}-loratgt{}.out.seed{}'.format(
          safe_model,
          lr,
          epoch_count,
          regularization_type,
          regularization_weight,
          lora_rank,
          sanitized_target,
          seed
          )
      config['resultsfile'] = 'gptjresults/{}-gender-lora-lr{:.2E}-epochs{}-regtype{}-regw{:.2E}-lorarank{}-loratgt{}.results.out.seed{}'.format(
          safe_model,
          lr,
          epoch_count,
          regularization_type,
          regularization_weight,
          lora_rank,
          sanitized_target,
          seed
          )
      with open('{}-lora.{}.sweep.yaml'.format(safe_model, i), 'w') as fout:
        yaml.dump(config, fout)

