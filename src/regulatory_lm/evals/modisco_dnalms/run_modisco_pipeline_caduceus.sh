#!/bin/bash

peak_file=$1
out_dir=$2
data_format=$3
ref_genome=$4
motif_db=$5

python -m regulatory_lm.evals.modisco_dnalms.modisco_predict_probs_caduceus --peak_file $peak_file --out_dir $out_dir --data_format $data_format --genome_fa $ref_genome
    
cd $out_dir

modisco motifs -s seqs.npz -a norm_probs.npz -n 1000000 -o modisco_results.h5 -w 350
modisco report -i modisco_results.h5 -o report/ -s report/ -m $motif_db