#! /bin/bash

__conda_setup="$('/u/scr/johnhew/miniconda3/' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/scr/johnhew/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/scr/johnhew/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/u/scr/johnhew/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

source ~/switch-cuda/switch-cuda.sh 11.7
conda activate pt2env

python ft_experiment.py $1 2> /dev/null
#python ft_experiment.py $1
