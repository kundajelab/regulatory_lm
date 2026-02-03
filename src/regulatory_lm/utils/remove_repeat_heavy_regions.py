import argparse
import pandas as pd
import numpy as np
import pyfaidx

def parse_args():
	parser = argparse.ArgumentParser(description="Given a peak set, removes all sequences which have a contiguous stretch of lowercase letters")
	parser.add_argument("--peak_file", type=str, help="List of peaks")
	parser.add_argument("--stretch_len", type=int, help="Length of lowercase stretch to remove", default=50)
	parser.add_argument("--seq_len", type=int, help="Length of sequences to test", default=350)
	parser.add_argument("--out_file", type=str, help="Output file")
	parser.add_argument("--data_format", type=str, choices=["bed", "narrowpeak"], help="Data format: 'bed' or 'narrowpeak'")
	parser.add_argument("--genome", type=str, help="Reference genome", default="/oak/stanford/groups/akundaje/patelas/regulatory_lm/data/hg38_repeat_lowercase.fa")
	args = parser.parse_args()
	return args

def has_lowercase_stretch(s: str, length: int) -> bool:
	'''
	Detects lowercase stretches in a sequence
	'''
	count = 0
	for c in s:
		if c.islower():
			count += 1
			if count >= length:
				return True
		else:
			count = 0
	return False

def subset_table(peak_table, stretch_len, seq_len, data_format, genome):
	'''
	Takes in a table of peaks/regions and subsets them to only contain the regions without repeat stretches of a certain length
	'''
	to_keep = []
	for seq in range(len(peak_table)):
		chrom = peak_table.loc[seq, 0]
		if data_format == "bed":
			orig_start, orig_end = peak_table.loc[seq, 1], peak_table.loc[seq, 2]
			midpoint = (orig_start + orig_end) // 2
		elif data_format == "narrowpeak":
			midpoint = peak_table.loc[seq, 1] + peak_table.loc[seq, 9]
		start = midpoint - seq_len // 2
		end = midpoint + seq_len // 2
		try:
			seq_str = genome[chrom][start:end].seq #In case there is a chromosome mismatch in the reference
		except:
			continue
		if not has_lowercase_stretch(seq_str, stretch_len):
			to_keep.append(seq)

	return peak_table.loc[to_keep]

def main():
	args = parse_args()
	peak_table = pd.read_csv(args.peak_file, sep="\t", header=None)
	genome = pyfaidx.Fasta(args.genome)
	table_subset = subset_table(peak_table, args.stretch_len, args.seq_len, args.data_format, genome)
	table_subset.to_csv(args.out_file, sep="\t", header=False, index=False)

if __name__ == "__main__":
	main()

