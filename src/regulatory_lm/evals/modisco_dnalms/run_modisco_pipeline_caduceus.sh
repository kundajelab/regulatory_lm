#!/bin/bash

# Source environment modules setup
source /etc/profile.d/modules.sh

# Now you can use "module"!
module load meme


peak_file=$1
out_dir=$2
data_format=$3

python -m regulatory_lm.evals.modisco_dnalms.modisco_predict_probs_caduceus --peak_file $peak_file --out_dir $out_dir --data_format $data_format
    
cd $out_dir

modisco motifs -s seqs.npz -a norm_probs.npz -n 1000000 -o modisco_results.h5 -w 350
modisco report -i modisco_results.h5 -o report/ -s report/ -m /users/patelas/scratch/H12CORE_meme_format.meme