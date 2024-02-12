#!/bin/bash

# source /sailhome/loraxie/.bashrc
source /nlp/scr/loraxie/miniconda3/etc/profile.d/conda.sh
/nlp/scr/loraxie/miniconda3/bin/conda init bash

conda activate backpack
python country_capital_hard_neg.py gpt-4