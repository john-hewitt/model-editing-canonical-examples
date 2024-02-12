
# small models
for size in 70m 160m 410m 1b; do for league in 0.001 0.0001 1e-05; do for k in full lora; do for j in country company gender stereoset verb temporal; do for i in `seq 0 4`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29 --gres gpu:1 --mem 30G   slurm_scripts/bpft_expt.sh configs/test/test-EleutherAI-pythia-$size-$j-$k-$league-$i.yaml"
  echo $cmd >> run.commands
  eval $cmd
done; done; done; done; done

## larger
#for size in 1.4b 2.8b; do for league in 0.001 0.0001 1e-05; do for k in full lora; do for j in country company gender stereoset verb temporal; do for i in `seq 0 4`; do 
#  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31 --gres gpu:1 --mem 30G   slurm_scripts/bpft_expt.sh configs/test/test-EleutherAI-pythia-$size-$j-$k-$league-$i.yaml"
#  echo $cmd >> run.commands
#  eval $cmd
#done; done; done; done; done
#
## largest for LoRA fits
#for size in 6.9b; do for league in 0.001 0.0001 1e-05; do for k in lora; do for j in country company gender stereoset verb temporal; do for i in `seq 0 4`; do 
#  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31,jagupard32,jagupard33 --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/test/test-EleutherAI-pythia-$size-$j-$k-$league-$i.yaml"
#  echo $cmd >> run.commands
#  eval $cmd
#done; done; done; done; done
#
## largest for full only fits on sphinx
#for size in 6.9b; do for league in 0.001 0.0001 1e-05; do for k in full; do for j in country company gender stereoset verb temporal; do for i in `seq 0 4`; do 
#  cmd="sbatch --account nlp --partition sphinx-lo --nodelist sphinx8 --gres gpu:1 --mem 60G   slurm_scripts/bpft_expt.sh configs/test/test-EleutherAI-pythia-$size-$j-$k-$league-$i.yaml"
#  echo $cmd >> run.commands
#  eval $cmd
#done; done; done; done; done
