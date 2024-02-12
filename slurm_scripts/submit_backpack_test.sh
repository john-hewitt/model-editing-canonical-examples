 for league in 0.001 0.0001 1e-05; do for k in full senses lora; do for j in company country temporal stereoset gender verb; do for i in `seq 0 9`; do 
  cmd="sbatch --account nlp --partition jag-lo --exclude jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25 --gres gpu:1 --mem 30G   slurm_scripts/bpft_expt.sh configs/test/test-stanfordnlp-backpack-gpt2-$j-$k-$league-$i.yaml"
  echo $cmd >> backpack.test.commands
  eval $cmd
done; done; done; done

