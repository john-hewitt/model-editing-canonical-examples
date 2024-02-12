import json

# format: {target: [list_of_bias_answers]}

def make_target_to_bias(in_path, out_path):
    target_to_bias = {}
    with open (in_path, 'r') as fin:
        for line in fin:
            line = json.loads(line)
            target = line['target']
            bias_answer = line['bias_answer']
            if target not in target_to_bias:
                target_to_bias[target] = [bias_answer]
            elif bias_answer not in target_to_bias[target]:
                target_to_bias[target].append(bias_answer)

    # write to file
    with open(out_path, 'w') as fout:
        for target, bias_answers in target_to_bias.items():
            data = {
                'target': target,
                'bias_answer': bias_answers
            }
            fout.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    make_target_to_bias("split/stereoset_train-val.jsonl", "target_to_bias-val.jsonl")
    make_target_to_bias("split/stereoset_train-test.jsonl", "target_to_bias-test.jsonl") 
