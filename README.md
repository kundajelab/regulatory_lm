# ARSENAL

This repo contains all code for the ARSENAL language modeling project. 

Pretrained models and relevant data not from other publications can be found at https://www.synapse.org/Synapse:syn72351987/wiki/ 

To set up a conda environment for this repo, run `bash setup_env.sh [ENV NAME]`

# Model Training
To train an ARSENAL model, you will need a config yml file. Examples can be found in the src/regulatory_lm/config/ folder. This file should contain all the relevant parameters for the model's embedder, encoder, and decoder modules, relevant training parameters (learning rate, number of epochs, mask probability, etc...), and relevant data files (training dataset, reference genome, etc...)

You can view the modeling options in src/regulatory_lm/modeling/model.py, and an exhaustive list of the parameters used in the relevant training script. 

To train an ARSENAL model, navigate to the src/ folder and run the following command:

`python regulatory_lm.modeling.train_peaks_with_repeat_suppression_and_fourier_loss [PATH_TO_CONFIG]`

If you'd like to train a model without the Fourier loss function for comparison, you can run this command instead:

`python regulatory_lm.modeling.train_peaks_with_repeat_suppression.py [PATH_TO_CONFIG]`

# Important Notebooks
We provide notebooks for important use cases of the ARSENAL model

`notebooks/regulatory_region_analysis.ipynb` - runs visualization and nucleotide dependency analyses for supplied regulatory regions

`notebooks/guided_generation.ipynb` - runs supervised model-guided sequence generation as demonstrated in the paper. Can easily be extended to other use cases and objectives. 

`notebooks/supervised_variant_scoring_african.ipynb` and `notebooks/supervised_variant_scoring_yoruban.ipynb` - runs statistics on supervised variant scores (see below) - requires ground truth scores from [DART-EVAL](https://github.com/kundajelab/DART-Eval).


# Downstream Supervised Models
To apply ARSENAL embeddings to train a downstream ChromBPNet model, [this repo](https://github.com/amanpatel101/arsenal-chrombpnet) should be installed (probably in its own environment). 

First, run `export ARSENAL_MODEL_DIR=[PATH TO ARSENAL REPO]`

To train an ARSENAL+ChromBPNet model, run the following command: `chrombpnet train --model_type arsenal-chrombpnet --out_dir [OUTPUT DIR] --input_embedding_dim 768 --arsenal_output_type embedding --peaks [PEAK FILE] --negatives [NEGATIVE FILE] --bigwig [BIGWIG FILE] --bias [BIAS MODEL FILE] --fasta [REFERENCE GENOME] --chrom_sizes [CHROM SIZES FILE] --arsenal_model [ARSENAL MODEL .PTH FILE] --arsenal_input_size 350 --num_layers_avg [LAST N EMBEDDING LAYERS TO AVERAGE]`

To score variants using this trained model, run the following command: `snp_score -l [VARIANT LIST] -g [REFERENCE GENOME] -s [CHROM SIZES FILE] --model_type arsenal-chrombpnet --model [BEST MODEL .ckpt FILE] --out_prefix [OUTPUT PREFIX/DIR] --total_shuf 2`

To train a regular ChromBPNet model for comparison, run the following command: `chrombpnet train --model_type chrombpnet --out_dir [OUTPUT DIR] --peaks [PEAK FILE] --negatives [NEGATIVE FILE] --bigwig [BIGWIG FILE] --bias [BIAS MODEL FILE] --fasta [REFERENCE GENOME] --chrom_sizes [CHROM SIZES FILE]`

To score variants using this trained model, run the following command: `snp_score -l [VARIANT LIST] -g [REFERENCE GENOME] -s [CHROM SIZES FILE] --model_type chrombpnet --model [BEST MODEL .pt FILE] --out_prefix [OUTPUT PREFIX/DIR] --total_shuf 2`

# TF-MoDISco Analysis
To run TF-MoDISco analysis on ARSENAL models, navigate to `src/` and run the following command: `bash regulatory_lm/evals/run_modisco_pipeline.sh [PEAK FILE] [ARSENAL MODEL DIR] [CHECKPOINT NUMBER] [OUTPUT DIR] [DATA FORMAT (bed or narrowpeak)] [REFERENCE GENOME] [MEME MOTIF DB]`

# DART-EVAL Benchmarking
We include benchmarking on two zero-shot [DART-EVAL](https://github.com/kundajelab/DART-Eval) tasks in the ARSENAL paper. Code to run these tasks exists in the `regulatory_lm` branch of that repo. Note that you will likely need to install the `rotary-embedding-torch` package to the environment you use for DART-EVAL.  

# Analysis With Other DNALMs
We include results from Caduceus and HyenaDNA in the paper. 

The notebook `notebooks/nuc_deps_other_models.ipynb` allows for visualization of regulatory regions using these two models. 

To run TF-MoDISco using these models, you can run the following commands:

`bash regulatory_lm/evals/modisco_dnalms/run_modisco_pipeline_hyena.sh [PEAK FILE] [OUTPUT DIR] [DATA FORMAT (bed or narrowpeak)] [REFERENCE GENOME] [MEME MOTIF DB]`

`bash regulatory_lm/evals/modisco_dnalms/run_modisco_pipeline_caduceus.sh [PEAK FILE] [OUTPUT DIR] [DATA FORMAT (bed or narrowpeak)] [REFERENCE GENOME] [MEME MOTIF DB]`


