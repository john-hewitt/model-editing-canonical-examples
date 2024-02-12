python convert_unsup.py > temporal_unconditional.jsonl
python convert_train.py > temporal_train.jsonl
python convert_eval.py > temporal_eval_clear.jsonl

mkdir split
python split_val_test.py
