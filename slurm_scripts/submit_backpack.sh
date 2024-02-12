 for k in lora sense full; do for j in country_capital company_ceo pronoun_gender_bias verb_conjugation temporal stereoset; do for i in `seq 0 24`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25 --gres gpu:1 --mem 30G   slurm_scripts/bpft_expt.sh configs/${j}/backpack_sweep/stanfordnlp-backpack-gpt2-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done
