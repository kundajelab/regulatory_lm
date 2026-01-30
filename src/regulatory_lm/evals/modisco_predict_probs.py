import os
import torch
import sys
from ..modeling.model import *
from torch.utils.data import DataLoader, Subset
from ..dataloader.data_generator_peaks import NarrowpeakDataset, BedDataset
import torch.nn.functional as F
import numpy as np
import argparse
from ..modeling.utils import load_model
import yaml
from tqdm import tqdm
import torch._dynamo
import json


torch._dynamo.config.suppress_errors = True

def parse_args():
	parser = argparse.ArgumentParser(description="Predicts probabilities for modisco given a model and set of peaks")
	parser.add_argument("--peak_file", type=str, help="List of regions to predict on")
	parser.add_argument("--model_dir", type=str, help="Model directory")
	parser.add_argument("--checkpoint", type=int, help="Model checkpoint")
	parser.add_argument("--genome_fa", type=str, help="Genome Fasta")
	parser.add_argument("--out_dir", type=str, help="Output directory")
	parser.add_argument("--data_format", type=str, help="Format of peak file", choices=["bed", "narrowpeak"], default="narrowpeak")
	parser.add_argument("--batch_size", type=int, help="Batch size", default=1024)
	args = parser.parse_args()
	return args


def predict_probs(model, dataloader, num_real_tokens, out_dir, device, optimizer=None):
	# model, _, _, _ = load_model(model, None, state_path)
	model.eval()
	model.to(device)
	
	probs_norm_lst = []
	seqs_lst = []
	
	for i, seqs in enumerate(tqdm(dataloader, ncols=100, unit="batch")): #assumes motif_mask is bool tensor
		assert optimizer is None
		seqs = seqs.to(device, dtype=torch.long)
		
		_, seq_len = seqs.size()
		masked_probs = []

		for pos in range(seq_len):
			masked_seqs = seqs.clone()
			mask_token_id = num_real_tokens + 1
			
			masked_seqs[:, pos] = mask_token_id  # Mask the nucleotide at position `pos`
			with torch.no_grad():
				logits = model(masked_seqs, None)
				probs = F.softmax(logits, dim=-1).to(dtype=torch.float32).permute(0,2,1)
				probs_norm = probs.cpu()
			
			masked_probs.append(probs_norm[:, :, pos])  # Extract the probability for the masked position
		for i, ind in enumerate(list(range(seq_len))):
			probs_norm[:,:,ind] = masked_probs[i]
		probs_norm = probs_norm.permute(0,2,1)
		nuc_average = torch.mean(probs_norm, dim=1).unsqueeze(1)
		# nuc_average_expanded = nuc_average.unsqueeze(2)
		normalized = probs_norm / nuc_average
		epsilon = 1e-10
		normalized = normalized + (normalized == 0).to(dtype=torch.float32) * epsilon
		probs_norm = (probs_norm * torch.log(normalized)).to(dtype=torch.float32).cpu().numpy(force=True)

		# masked_probs = np.stack(masked_probs, axis=1)  # Stack along sequence length
		
		del logits, probs, nuc_average#, nuc_average_expanded
		torch.cuda.empty_cache()
		
		probs_norm_lst.append(probs_norm)
		
		one_hot = torch.zeros(seqs.shape[0], seqs.shape[1], 4, dtype=torch.int8)
		for nuc in range(4):
			one_hot[:, :, nuc] = (seqs == nuc).to(dtype=torch.int8)  # for non ACGT, set to 0
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
	model_dir = script_args.model_dir
	checkpoint = script_args.checkpoint
	output_dir = script_args.out_dir

	args_path = os.path.join(model_dir, "args.json")
	with open(args_path, "r") as f:
		args = json.load(f)

	embedder_kwargs = args.get("embedder_kwargs", {})
	encoder_kwargs = args.get("encoder_kwargs", {})
	decoder_kwargs = args.get("decoder_kwargs", {})
	model_kwargs = args.get("model_kwargs", {})

	checkpoint_path = os.path.join(model_dir, f"checkpoint_{checkpoint}.pt")

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
	
	embedder = MODULES[args["embedder"]](
		args["embedding_size"], vocab_size=args["num_real_tokens"] + 2, masking=True, **embedder_kwargs
	)
	encoder = MODULES[args["encoder"]](
		args["embedding_size"], args["num_encoder_layers"], **encoder_kwargs
	)
	decoder = MODULES[args["decoder"]](
		args["embedding_size"], **decoder_kwargs
	)
	model = RegulatoryLM(embedder, encoder, decoder, **model_kwargs)

	float_dtype = FLOAT_DTYPES[args["float_dtype"]]
	if device == torch.device("cuda"):
		model = model.to(device, dtype=float_dtype)

	model_info = torch.load(checkpoint_path)
	if list(model_info["model_state"].keys())[0][:7] == "module.":
		print("here")
		model_info["model_state"] = {x[7:]:model_info["model_state"][x] for x in model_info["model_state"]}
	else:
		model = torch.compile(model)
	model.load_state_dict(model_info["model_state"])
	predict_probs(model, test_loader, args["num_real_tokens"], output_dir, device, optimizer=None)

if __name__ == "__main__":
	main()