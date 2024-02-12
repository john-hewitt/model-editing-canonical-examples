
import sys
import random
import os


dataset_lines = []

for filepath in sys.argv[1:]:
  dataset_lines.append([line for line in open(filepath)])


length = len(dataset_lines[0])
for d in dataset_lines:
  assert len(d) == length

# random.seed(hash('pizza'))
random.seed(888)
indices = [x for x in range(length)]
random.shuffle(indices)

for i, filepath in enumerate(sys.argv[1:]):
  with open('split/{}-val.jsonl'.format(os.path.basename(filepath).replace('.jsonl', '')), 'w') as fout:
    for index in indices[:length//2]:
      fout.write(dataset_lines[i][index])
  with open('split/{}-test.jsonl'.format(os.path.basename(filepath).replace('.jsonl', '')), 'w') as fout:
    for index in indices[length//2:]:
      fout.write(dataset_lines[i][index])
