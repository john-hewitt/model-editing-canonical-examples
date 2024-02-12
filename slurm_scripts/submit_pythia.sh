# modest-sized models
for size in 70m 160m 410m 1b; do for k in full lora; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25    --gres gpu:1 --mem 30G   slurm_scripts/bpft_expt.sh configs/${j}/pythia_sweep/EleutherAI-pythia-$size-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done; done

# large models
for size in 1.4b 2.8b; do for k in lora full; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31  --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/${j}/pythia_sweep/EleutherAI-pythia-$size-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done; done

for size in 6.9b; do for k in lora; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31  --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/${j}/pythia_sweep/EleutherAI-pythia-$size-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done; done

for size in 6.9b; do for k in full; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition sphinx-lo --nodelist sphinx8  --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/${j}/pythia_sweep/EleutherAI-pythia-$size-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done; done
