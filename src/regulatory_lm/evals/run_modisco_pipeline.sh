#!/bin/bash

peak_file=$1
model_dir=$2
checkpoint=$3
out_dir=$4
data_format=$5
ref_genome=$6
motif_db=$7

python -m regulatory_lm.evals.modisco_predict_probs --peak_file $peak_file --model_dir $model_dir --checkpoint $checkpoint --out_dir $out_dir --data_format $data_format --genome_fa $ref_genome
    
cd $out_dir

modisco motifs -s seqs.npz -a norm_probs.npz -n 1000000 -o modisco_results.h5 -w 350
modisco report -i modisco_results.h5 -o report/ -s report/ -m $motif_db