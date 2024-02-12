# large models
 for league in 1e-05 0.0001 0.001; do for k in senses; do for j in company country verb gender temporal stereoset; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31 --gres gpu:1 --mem 60G   slurm_scripts/imp_expt.sh configs/test/test-stanfordnlp-backpack-gpt2-$j-$k-$league-$i.yaml"
  echo $cmd >> backpack.test.commands
  eval $cmd
done; done; done; done

# for league in 0.001 0.0001 1e-05; do for k in senses; do for j in company country verb gender temporal stereoset; do for i in `seq 0 9`; do 
#  cmd="sbatch --account nlp --partition sphinx --nodelist sphinx1 --gres gpu:1 --mem 60G   slurm_scripts/imp_expt.sh configs/test/test-stanfordnlp-backpack-gpt2-$j-$k-$league-$i.yaml"
#  echo $cmd >> backpack.test.commands
#  eval $cmd
#done; done; done; done
