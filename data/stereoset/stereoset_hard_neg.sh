#!/bin/bash

# source /sailhome/loraxie/.bashrc
source /nlp/scr/loraxie/miniconda3/etc/profile.d/conda.sh
/nlp/scr/loraxie/miniconda3/bin/conda init bash

conda activate backpack
python target_to_bias.py
python make_stereoset_hard_neg.py new gpt-4
python make_stereoset_hard_neg.py sentence gpt-4