for k in lora; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31  --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/${j}/gptj_sweep/EleutherAI-gpt-j-6b-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done

for k in full; do for j in country_capital temporal company_ceo verb_conjugation stereoset pronoun_gender_bias; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition sphinx-lo --nodelist sphinx8  --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/${j}/gptj_sweep/EleutherAI-gpt-j-6b-$k.$i.sweep.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done
