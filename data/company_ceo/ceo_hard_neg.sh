#!/bin/bash

# source /sailhome/loraxie/.bashrc
source /nlp/scr/loraxie/miniconda3/etc/profile.d/conda.sh
/nlp/scr/loraxie/miniconda3/bin/conda init bash

conda activate backpack
python ceo_data.py hard_neg_companies_and_ceos.jsonl company_ceo_multi_hard_neg.jsonl gpt-4
python make_splits_hard_neg.py