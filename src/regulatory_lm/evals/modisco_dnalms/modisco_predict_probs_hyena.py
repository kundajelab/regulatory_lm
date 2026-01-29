import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from ...modeling.model import *
from torch.utils.data import DataLoader, Subset
from ...dataloader.data_generator_peaks import NarrowpeakDataset, BedDataset
import torch.nn.functional as F
import numpy as np
import argparse
from ...modeling.utils import load_model
import yaml
from tqdm import tqdm
import torch._dynamo
import json


torch._dynamo.config.suppress_errors = True

def parse_args():
	parser = argparse.ArgumentParser(description="Predicts probabilities for modisco given a model and set of peaks")
	parser.add_argument("--peak_file", type=str, help="List of regions to predict on")
	parser.add_argument("--genome_fa", type=str, help="Genome Fasta", default="/mnt/lab_data2/regulatory_lm/oak_backup/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
	parser.add_argument("--out_dir", type=str, help="Output directory")
	parser.add_argument("--data_format", type=str, help="Format of peak file", choices=["bed", "narrowpeak"], default="narrowpeak")
	parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
	args = parser.parse_args()
	return args

def predict_probs(model, dataloader, out_dir, device, optimizer=None):
	# model, _, _, _ = load_model(model, None, state_path)
	model.eval()
	model.to(device)
	
	probs_norm_lst = []
	seqs_lst = []
	
	for i, seqs in enumerate(tqdm(dataloader, ncols=100, unit="batch")): #assumes motif_mask is bool tensor        
		assert optimizer is None
		seqs = seqs.to(device, dtype=torch.long)
		seqs = seqs + 7 #convert tokenization
		_, seq_len = seqs.size()
		masked_probs = []

		with torch.no_grad():
			logits = model(seqs).logits
			probs = F.softmax(logits, dim=-1)[:,:,7:11]
		nuc_average = torch.mean(probs, dim=1).unsqueeze(1)
		# nuc_average_expanded = nuc_average.unsqueeze(2)
		normalized = probs / nuc_average
		epsilon = 1e-10
		normalized = normalized + (normalized == 0).to(dtype=torch.float32) * epsilon
		probs_norm = (probs * torch.log(normalized)).to(dtype=torch.float32).cpu().numpy(force=True)
		
		
		del logits, probs, nuc_average#, nuc_average_expanded
		torch.cuda.empty_cache()
		
		probs_norm_lst.append(probs_norm)
		
		one_hot = torch.zeros(seqs.shape[0], seqs.shape[1], 4, dtype=torch.int8)
		for nuc in range(4):
			one_hot[:, :, nuc] = (seqs - 7 == nuc).to(dtype=torch.int8)  # for non ACGT, set to 0
		one_hot = one_hot.cpu().numpy(force=True)
		
		del seqs
		torch.cuda.empty_cache()
		
		seqs_lst.append(one_hot)
	
	norm_probs = np.concatenate(probs_norm_lst, axis=0)
	seqs = np.concatenate(seqs_lst, axis=0)
	
	os.makedirs(out_dir, exist_ok=True)
	norm_probs_path = os.path.join(out_dir, "norm_probs.npz")
	seqs_path = os.path.join(out_dir, "seqs.npz")
	
	seqs = np.transpose(seqs, (0, 2, 1))
	norm_probs = np.transpose(norm_probs, (0, 2, 1))
	
	np.savez_compressed(norm_probs_path, norm_probs)
	np.savez_compressed(seqs_path, seqs)
	

def main():
	args = parse_args
def main():
	script_args = parse_args()
	output_dir = script_args.out_dir

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	FLOAT_DTYPES = {
		"float32": torch.float32,
		"float64": torch.float64,
		"bfloat16": torch.bfloat16,
		"float16": torch.float16
	}

	chrom_list = ["chr" + str(x) for x in range(1,23)] + ["chrX", "chrY"]
	if script_args.data_format == "narrowpeak":
		print("Data type: narrowpeak")
		test_dataset = NarrowpeakDataset(script_args.peak_file, script_args.genome_fa, chrom_list, mode="test")
	elif script_args.data_format == "bed":
		print("Data type: bed")
		test_dataset = BedDataset(script_args.peak_file, script_args.genome_fa, chrom_list, mode="test")
	test_loader = DataLoader(test_dataset, batch_size=script_args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
	
	model_name = f"LongSafari/hyenadna-large-1m-seqlen-hf"
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
	model =  AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

	predict_probs(model, test_loader, output_dir, device, optimizer=None)

if __name__ == "__main__":
	main()