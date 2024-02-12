"""Get conjugation data"""

split_dir_name = 'split_redo'
verb_picking_strategy = 'hard' # 'random'
max_templates_per_mlset = 16

train_syneval_subsets = ['simple_agrmt', 'vp_coord']
eval_syneval_subsets = ['long_vp_coord', 'sent_comp', 'subj_rel', ]

num_verbs_to_use = 100 # for random; these will be chosen randomly
verb_list_path = "combined_verb_list.csv"
syneval_template_path = "LM_syneval/data/templates"

make_prefix_pool = False
prefix_to_add = "." # None



import pickle
import transformers
import pandas as pd 
import json
import numpy as np

import os
os.makedirs(split_dir_name, exist_ok=True)

np.random.seed(888)

# if desired, get a random pool of prefixes from webtext
random_prefixes = []
if make_prefix_pool:
    from datasets import load_dataset
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
    shuffled_dataset = dataset.shuffle(buffer_size=10_000, seed=42)
    dataset_iter = iter(shuffled_dataset)
    for i in range(1000):
        next_chunk = next(dataset_iter)
        vals = next_chunk['text'].split('.')
        if len(vals) > 2:
            cur_sent = vals[2].strip()
            if len(cur_sent) > 30 and len(cur_sent) < 120:
                random_prefixes.append(cur_sent + '.')


def load_verbs(tokenizer, verb_list_path):
    """Load a list of verbs from refining-tse paper, and filter to consider 
    only verbs where both sg and pl are a single token"""
    verb_df = pd.read_csv(verb_list_path)
    raw_sg_verbs, raw_pl_verbs = verb_df['sing'].tolist(), verb_df['plur'].tolist()
    raw_sg_verbs = [' ' + x for x in raw_sg_verbs]
    raw_pl_verbs = [' ' + x for x in raw_pl_verbs]

    # filter to single-token verbs
    sg_verbs = []
    pl_verbs = []
    for i in range(len(raw_sg_verbs)):
        sg_verb_id = tokenizer.encode(raw_sg_verbs[i])
        pl_verb_id = tokenizer.encode(raw_pl_verbs[i])
        if len(sg_verb_id) == 1 and \
            len(pl_verb_id) == 1:
            sg_verbs.append(raw_sg_verbs[i])
            pl_verbs.append(raw_pl_verbs[i])
            
    return raw_sg_verbs, raw_pl_verbs, sg_verbs, pl_verbs

def _extract_syneval_sample_left(
        tokenizer, string_tup, 
        raw_sg_verbs, raw_pl_verbs,
        verbose=False,
    ):
    """Extract a syneval dataset string tuple into an example in the min pair format"""

    # cut off the leading "the"
    assert string_tup[0][:3].lower() == 'the'
    s0 = ' The' + string_tup[0][3:]
    s1 = ' The' + string_tup[1][3:]

    corr_tokens = tokenizer.encode(s0)
    incorr_tokens = tokenizer.encode(s1)

    # find the first index of a token mismatch
    mismatch_index = -1 
    for i in range(min(len(corr_tokens), len(incorr_tokens))):
        if corr_tokens[i] != incorr_tokens[i]:
            mismatch_index = i
            break 

    if mismatch_index == -1:
        # catch special case of no mismatch found due to non-single-token verb
        assert 'swim' in string_tup[0], f"no mismatch found in {string_tup}"
        mismatch_index = min(len(corr_tokens), len(incorr_tokens)) - 1
        left_context_tokens = corr_tokens[:mismatch_index]        
        label = 'sg' if len(corr_tokens) > len(incorr_tokens) else 'pl'
    else:
        left_context_tokens = corr_tokens[:mismatch_index]
        
        target_word = tokenizer.decode(corr_tokens[mismatch_index])
        contrast_word = tokenizer.decode(incorr_tokens[mismatch_index])
        if target_word in raw_sg_verbs and contrast_word in raw_pl_verbs:
            label = 'sg'
        elif target_word in raw_pl_verbs and contrast_word in raw_sg_verbs:
            label = 'pl'
        else:
            print('warning: unknown', target_word, contrast_word)
            label = 'unknown'

    left_str = tokenizer.decode(left_context_tokens)
    return left_str, label

def get_min_pair_dataset(tokenizer, syneval_dict, raw_sg_verbs, raw_pl_verbs):
    # catch exception
    raw_sg_verbs += [' smiles']
    raw_pl_verbs += [' smile']


    # get the left context of everything, as well as the label of sg/pl
    raw_min_pair_dataset = []
    for k in syneval_dict:
        for ex in syneval_dict[k]:
            left_string, label = _extract_syneval_sample_left(
                tokenizer, ex, 
                raw_sg_verbs, raw_pl_verbs)
            if label != 'unknown':
                raw_min_pair_dataset.append( (left_string, label) )

    # deduplicate
    seen_examples = set()
    min_pair_dataset = []
    for left_string, label in raw_min_pair_dataset:
        min_pair_hash = left_string + label
        if min_pair_hash not in seen_examples:
            seen_examples.add(min_pair_hash)
            min_pair_dataset.append((left_string, label))
    return min_pair_dataset


def preprocess_syneval_subsets(syneval_subsets, raw_sg_verbs, raw_pl_verbs, max_templates_per_mlset=2, verbose=False):
    print("Preprocessing syneval subsets:", syneval_subsets)
    def _get_syneval_category(syneval_subset):
        syneval_dict = pickle.load(open(f"{syneval_template_path}/{syneval_subset}.pickle", 'rb'))
        min_pair_dataset = get_min_pair_dataset(
            tokenizer, syneval_dict, 
            raw_sg_verbs, raw_pl_verbs, 
        )

        selected_indices = np.random.choice(range(len(min_pair_dataset)), max_templates_per_mlset, replace=False)
        min_pair_dataset = [min_pair_dataset[i] for i in selected_indices]
        assert len(min_pair_dataset) == max_templates_per_mlset
        # min_pair_dataset contains (left_string, label)
        
        # separate the val and test sets
        val_indices = np.random.choice(range(len(min_pair_dataset)), len(min_pair_dataset) // 2, replace=False)

        if verbose:
            print("DEBUG min_pair_dataset", min_pair_dataset)
        val_min_pair_dataset = [min_pair_dataset[i] for i in val_indices]
        test_min_pair_dataset = [min_pair_dataset[i] for i in range(len(min_pair_dataset)) if i not in val_indices]
        if verbose:
            print("DEBUG len(val_min_pair_dataset), len(test_min_pair_dataset)", len(val_min_pair_dataset), len(test_min_pair_dataset))
            print("val_min_pair_dataset", val_min_pair_dataset)
            print("test_min_pair_dataset", test_min_pair_dataset)
            print()
        return val_min_pair_dataset, test_min_pair_dataset

    val_set = []
    test_set = []
    for syneval_subset in syneval_subsets:
        val_min_pair_dataset, test_min_pair_dataset = _get_syneval_category(syneval_subset)
        val_set += val_min_pair_dataset
        test_set += test_min_pair_dataset
    return val_set, test_set 

def load_data_examples(min_pair_dataset, prefix_to_add, sg_verbs, pl_verbs, invert=False):
    # load training examples from M and L and construct a train set 

    correct_examples = [] # correct demonstrations 

    # assemble completions for all verbs
    for left_string, label in min_pair_dataset:
        for i in range(len(sg_verbs)):
            verb_options = {'sg': sg_verbs[i], 'pl': pl_verbs[i]}
            if invert: 
                label_key = 'sg' if label == 'pl' else 'pl'
            else:
                label_key = label

            if prefix_to_add is not None:
                cur_prefix_to_add = prefix_to_add
            elif prefix_to_add is None and len(random_prefixes) > 0:
                # if not specified and there is a pool of random_prefixes, pick one
                cur_prefix_to_add = np.random.choice(random_prefixes)

            correct_examples.append(
                {
                    "prefix": cur_prefix_to_add + left_string,
                    "suffix": verb_options[label_key],
                    "suffix1": verb_options[label_key],
                    "suffix2": verb_options['sg' if label_key == 'pl' else 'pl'],
                }
            )
    return correct_examples

def dump_to_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for x in data:
            f.write(json.dumps(x) + '\n')

def get_hard_verbs(tokenizer, sg_verbs, pl_verbs):
    verbs_failed_on_simple_agrmt = " arms| auctions| bands| banks| bars| beats| bets| codes| courts| deserts| drums| franchises| guts| hurts| lasts| lets| levels| loans| masters| minds| parts| petitions| ramps| screens| shadows| shuts| stocks| taxes| teams| wounds"
    verbs_failed_on_simple_agrmt = verbs_failed_on_simple_agrmt.split('|')
    verb_indices = []
    for x in verbs_failed_on_simple_agrmt:
        if x in sg_verbs:
            verb_indices.append(sg_verbs.index(x))
        elif x in pl_verbs:
            verb_indices.append(pl_verbs.index(x))
        else:
            raise ValueError("Invalid verb")
    num_verbs_to_use = len(verb_indices)
    print(f"Found {num_verbs_to_use} hard verbs to use")
    val_verb_indices = np.random.choice(verb_indices, num_verbs_to_use // 2, replace=False)
    test_verb_indices = [x for x in verb_indices if x not in val_verb_indices]
    return val_verb_indices, test_verb_indices

def get_random_verbs(sg_verbs, num_verbs_to_use):
    verb_indices = np.random.choice(range(len(sg_verbs)), num_verbs_to_use, replace=False)
    val_verb_indices = np.random.choice(verb_indices, num_verbs_to_use // 2, replace=False)
    test_verb_indices = [x for x in verb_indices if x not in val_verb_indices]
    return val_verb_indices, test_verb_indices



if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    raw_sg_verbs, raw_pl_verbs, sg_verbs, pl_verbs = \
        load_verbs(tokenizer, verb_list_path=verb_list_path)

    if verb_picking_strategy == 'hard':
        val_verb_indices, test_verb_indices = get_hard_verbs(tokenizer, sg_verbs, pl_verbs)
    elif verb_picking_strategy == 'random':
        val_verb_indices, test_verb_indices = get_random_verbs(sg_verbs, num_verbs_to_use)
    else:
        raise ValueError("invalid verb_picking_strategy")
    

    train_and_eval_by_split = {}
    train_val_set, train_test_set = preprocess_syneval_subsets(train_syneval_subsets, raw_sg_verbs, raw_pl_verbs, max_templates_per_mlset=max_templates_per_mlset)
    eval_val_set, eval_test_set = preprocess_syneval_subsets(eval_syneval_subsets, raw_sg_verbs, raw_pl_verbs, max_templates_per_mlset=max_templates_per_mlset)
    train_and_eval_by_split["val"] = (train_val_set, eval_val_set)
    train_and_eval_by_split["test"] = (train_test_set, eval_test_set)

    for split_name, verb_indices in [("val", val_verb_indices), ("test", test_verb_indices)]:

        cur_sg_verbs = np.array(sg_verbs)[verb_indices]
        cur_pl_verbs = np.array(pl_verbs)[verb_indices]

        train_set, eval_set = train_and_eval_by_split[split_name]
    
        train_data = load_data_examples(
            train_set, prefix_to_add, 
            cur_sg_verbs, cur_pl_verbs)
        dump_to_jsonl(train_data, f"{split_dir_name}/verb_conjugation_train-{split_name}.jsonl")

        eval_data = load_data_examples(
            eval_set, prefix_to_add, 
            cur_sg_verbs, cur_pl_verbs)
        dump_to_jsonl(eval_data, f"{split_dir_name}/verb_conjugation_eval-{split_name}.jsonl")
        

        eval_data_text = [{'text': x['prefix'] + x['suffix']} for x in eval_data]
        dump_to_jsonl(eval_data_text, f"{split_dir_name}/verb_conjugation_eval_unconditional-{split_name}.jsonl")

