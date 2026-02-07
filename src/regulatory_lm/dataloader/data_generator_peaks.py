from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import pyfaidx
import torch
import random
import hashlib
from .utils import encode_sequence, rev_comp
import json

class NarrowpeakDataset(Dataset):
	def __init__(self, peak_file, genome_fa, chrom_list, mode="train", seq_len=350, rev_comp_prob=0.5, jitter=50):
		'''
		Dataset class for narrowpeak (10 column) data type
		Required parameters are:
		-peak_file: file name of list of peaks/regions
		-genome_fa: reference genome file
		-chrom_list: list of chromosomes to consider
		-mode: "train" or otherwise, decides whether reverse complementing and jittering are done
		-seq_len: sequence length to train on 
		-rev_comp_prob: probability of reverse complementing
		-jitter: max distance to shift the sequence
		'''
		self.seq_len = seq_len
		self.peak_file = peak_file
		self.genome = pyfaidx.Fasta(genome_fa, one_based_attributes=False, sequence_always_upper=True)
		self.mode = mode
		self._rev_comp_prob = rev_comp_prob
		self._jitter = jitter
		self.chrom_list = chrom_list
		self.regions = self.load_regions()

	def __len__(self):
		return len(self.regions)

	def load_regions(self):
		'''
		Loads a set of regions from a given input file
		'''
		chrom_set = set(self.chrom_list)
		peak_regions = []
		# for encid in os.listdir(self.data_dir):
		#     curr_filename = os.path.join(self.data_dir, encid, "preprocessing/downloads/peaks.bed.gz")
		#     if not os.path.exists(curr_filename):
		#         continue
		peak_data = pd.read_csv(self.peak_file, sep="\t", header=None)
		peak_data = peak_data.loc[peak_data[0].isin(chrom_set)].reset_index(drop=True)
		peak_data["true_start"] = peak_data[1] + peak_data[9] - self.seq_len // 2
		peak_data["true_end"] = peak_data["true_start"] + self.seq_len
		for reg in range(len(peak_data)):
			peak_regions.append((peak_data.loc[reg, 0], peak_data.loc[reg, "true_start"], peak_data.loc[reg, "true_end"]))

		return peak_regions



	def _apply_jitter(self, start):
		'''
		Randomly shifts the sequence by a certain amount
		'''
		jitter = random.randint(-self._jitter, self._jitter)
		start += jitter
		return start

	def __getitem__(self, idx):
		chrom, start, end = self.regions[idx]
		if self.mode == "train":
			curr_len = end - start
			start = max(0, self._apply_jitter(start))
			end = start + curr_len

		buffer = (self.seq_len - (end - start)) // 2
		dna_seq = self.genome[chrom][start-buffer:start-buffer+self.seq_len].seq
		seq = np.array(encode_sequence(dna_seq))
		if self.mode == "train" and random.random() < self._rev_comp_prob:
			seq = rev_comp(seq)
		return torch.from_numpy(seq)

class BedDataset(Dataset):
	def __init__(self, peak_file, genome_fa, chrom_list, mode="train", seq_len=350, rev_comp_prob=0.5, jitter=50):
		'''
		Dataset class for bed (3 column) data type
		Required parameters are:
		-peak_file: file name of list of peaks/regions
		-genome_fa: reference genome file
		-chrom_list: list of chromosomes to consider
		-mode: "train" or otherwise, decides whether reverse complementing and jittering are done
		-seq_len: sequence length to train on 
		-rev_comp_prob: probability of reverse complementing
		-jitter: max distance to shift the sequence
		'''
		self.seq_len = seq_len
		self.peak_file = peak_file
		self.genome = pyfaidx.Fasta(genome_fa, one_based_attributes=False, sequence_always_upper=True)
		self.mode = mode
		self._rev_comp_prob = rev_comp_prob
		self._jitter = jitter
		self.chrom_list = chrom_list
		self.regions = self.load_regions()

	def __len__(self):
		return len(self.regions)

	def load_regions(self):
		'''
		Loads a set of regions from a given input file
		'''

		chrom_set = set(self.chrom_list)
		# for encid in os.listdir(self.data_dir):
		#     curr_filename = os.path.join(self.data_dir, encid, "preprocessing/downloads/peaks.bed.gz")
		#     if not os.path.exists(curr_filename):
		#         continue
		peak_data = pd.read_csv(self.peak_file, sep="\t", header=None)
		peak_data = peak_data.loc[peak_data[0].isin(chrom_set)].reset_index(drop=True)

		return peak_data



	def _apply_jitter(self, start):
		'''
		Randomly shifts the sequence by a certain amount
		'''
		jitter = random.randint(-self._jitter, self._jitter)
		start += jitter
		return start

	def __getitem__(self, idx):
		chrom, start, end = self.regions.loc[idx, 0], self.regions.loc[idx, 1], self.regions.loc[idx, 2]
		if self.mode == "train":
			curr_len = end - start
			start = max(0, self._apply_jitter(start))
			end = start + curr_len

		buffer = (self.seq_len - (end - start)) // 2
		dna_seq = self.genome[chrom][start-buffer:start-buffer+self.seq_len].seq
		seq = np.array(encode_sequence(dna_seq))
		if self.mode == "train" and random.random() < self._rev_comp_prob:
			seq = rev_comp(seq)
		return torch.from_numpy(seq)

class NarrowpeakDatasetWithRepeatMasking(Dataset):
	def __init__(self, peak_file, genome_fa, chrom_list, mode="train", seq_len=350, rev_comp_prob=0.5, jitter=50, min_stretch=0):
		'''
		Dataset class for narrowpeak (10 column) data type with repeat masking
		Required parameters are:
		-peak_file: file name of list of peaks/regions
		-genome_fa: reference genome file
		-chrom_list: list of chromosomes to consider
		-mode: "train" or otherwise, decides whether reverse complementing and jittering are done
		-seq_len: sequence length to train on 
		-rev_comp_prob: probability of reverse complementing
		-jitter: max distance to shift the sequence
		-min_stretch: minimum length of consecutive repeats above which to mask
		'''
		self.seq_len = seq_len
		self.peak_file = peak_file
		self.genome = pyfaidx.Fasta(genome_fa, one_based_attributes=False)
		self.mode = mode
		self._rev_comp_prob = rev_comp_prob
		self._jitter = jitter
		self.chrom_list = chrom_list
		self.min_stretch = min_stretch
		self.regions = self.load_regions()

	def __len__(self):
		return len(self.regions)

	def load_regions(self):
		'''
		Loads a set of regions from a given input file
		'''
		chrom_set = set(self.chrom_list)
		peak_regions = []
		peak_data = pd.read_csv(self.peak_file, sep="\t", header=None)
		peak_data = peak_data.loc[peak_data[0].isin(chrom_set)].reset_index(drop=True)
		peak_data["true_start"] = peak_data[1] + peak_data[9] - self.seq_len // 2
		peak_data["true_end"] = peak_data["true_start"] + self.seq_len
		for reg in range(len(peak_data)):
			peak_regions.append((peak_data.loc[reg, 0], peak_data.loc[reg, "true_start"], peak_data.loc[reg, "true_end"]))

		return peak_regions

	def _apply_jitter(self, start):
		'''
		Randomly shifts the sequence by a certain amount
		'''
		jitter = random.randint(-self._jitter, self._jitter)
		start += jitter
		return start
	
	def get_stretch_repeat_mask(self, dna_seq):
		'''
		Given a sequence, this function searches for repeat stretches
		If it finds a consecutive stretch above a certain length, then it flags that stretch
		Returns a binary mask of the sequence length
		'''
		raw_repeats = np.array([int(nuc.islower()) for nuc in dna_seq])
		final_repeats = np.array([0] * len(raw_repeats))
		padded = np.concatenate(([0], raw_repeats, [0]))           # shape (N+2,)
		diff = np.diff(padded)                                     # shape (N+1,)

		# '1' indicates stretch start, '-1' indicates end (exclusive)
		starts = np.where(diff == 1)[0]
		ends = np.where(diff == -1)[0]

		for s, e in zip(starts, ends):
			stretch_len = e - s
			if stretch_len > self.min_stretch:
				final_repeats[s:e] = 1

		return list(final_repeats)


	def __getitem__(self, idx):
		'''
		Given a particular index in the dataset, produces a training sample
		The procedure includes: shifting the sequence (if necessary), taking the central 350bp, producing the repeat mask
		Finally, we reverse complement if necessary
		'''
		chrom, start, end = self.regions[idx]
		if self.mode == "train":
			curr_len = end - start
			start = max(0, self._apply_jitter(start))
			end = start + curr_len

		buffer = (self.seq_len - (end - start)) // 2
		dna_seq = self.genome[chrom][start-buffer:start-buffer+self.seq_len].seq
		if self.min_stretch == 0:
			is_repeat = [int(nuc.islower()) for nuc in dna_seq] #Gets repeat annotation
		else:
			is_repeat = self.get_stretch_repeat_mask(dna_seq)
		seq = np.array(encode_sequence(dna_seq.upper()))
		if self.mode == "train" and random.random() < self._rev_comp_prob:
			seq = rev_comp(seq)
			is_repeat = is_repeat[::-1]
		return torch.from_numpy(seq), torch.tensor(is_repeat)


class BedDatasetWithRepeatMasking(Dataset):
	def __init__(self, peak_file, genome_fa, chrom_list, mode="train", seq_len=350, rev_comp_prob=0.5, jitter=50, min_stretch=0):
		'''
		Dataset class for bed (3 column) data type with repeat masking
		Required parameters are:
		-peak_file: file name of list of peaks/regions
		-genome_fa: reference genome file
		-chrom_list: list of chromosomes to consider
		-mode: "train" or otherwise, decides whether reverse complementing and jittering are done
		-seq_len: sequence length to train on 
		-rev_comp_prob: probability of reverse complementing
		-jitter: max distance to shift the sequence
		-min_stretch: minimum length of consecutive repeats above which to mask
		'''
		self.seq_len = seq_len
		self.peak_file = peak_file
		self.genome = pyfaidx.Fasta(genome_fa, one_based_attributes=False)
		self.mode = mode
		self._rev_comp_prob = rev_comp_prob
		self._jitter = jitter
		self.chrom_list = chrom_list
		self.min_stretch = min_stretch
		self.regions = self.load_regions()

	def __len__(self):
		return len(self.regions)

	def load_regions(self):
		'''
		Loads a set of regions from a given input file
		'''
		chrom_set = set(self.chrom_list)
		peak_data = pd.read_csv(self.peak_file, sep="\t", header=None)
		peak_data = peak_data.loc[peak_data[0].isin(chrom_set)].reset_index(drop=True)

		return peak_data

	def _apply_jitter(self, start):
		'''
		Randomly shifts the sequence by a certain amount
		'''
		jitter = random.randint(-self._jitter, self._jitter)
		start += jitter
		return start
	
	def get_stretch_repeat_mask(self, dna_seq):
		'''
		Given a sequence, this function searches for repeat stretches
		If it finds a consecutive stretch above a certain length, then it flags that stretch
		Returns a binary mask of the sequence length
		'''
		raw_repeats = np.array([int(nuc.islower()) for nuc in dna_seq])
		final_repeats = np.array([0] * len(raw_repeats))
		padded = np.concatenate(([0], raw_repeats, [0]))           # shape (N+2,)
		diff = np.diff(padded)                                     # shape (N+1,)

		# '1' indicates stretch start, '-1' indicates end (exclusive)
		starts = np.where(diff == 1)[0]
		ends = np.where(diff == -1)[0]

		for s, e in zip(starts, ends):
			stretch_len = e - s
			if stretch_len > self.min_stretch:
				final_repeats[s:e] = 1

		return list(final_repeats)


	def __getitem__(self, idx):
		'''
		Given a particular index in the dataset, produces a training sample
		The procedure includes: shifting the sequence (if necessary), taking the central 350bp, producing the repeat mask
		Finally, we reverse complement if necessary
		'''
		chrom, start, end = self.regions.loc[idx, 0], self.regions.loc[idx, 1], self.regions.loc[idx, 2]
		if self.mode == "train":
			curr_len = end - start
			start = max(0, self._apply_jitter(start))
			end = start + curr_len

		buffer = (self.seq_len - (end - start)) // 2
		dna_seq = self.genome[chrom][start-buffer:start-buffer+self.seq_len].seq
		if self.min_stretch == 0:
			is_repeat = [int(nuc.islower()) for nuc in dna_seq] #Gets repeat annotation
		else:
			is_repeat = self.get_stretch_repeat_mask(dna_seq)
		seq = np.array(encode_sequence(dna_seq.upper()))
		if self.mode == "train" and random.random() < self._rev_comp_prob:
			seq = rev_comp(seq)
			is_repeat = is_repeat[::-1]
		return torch.from_numpy(seq), torch.tensor(is_repeat)


class BedDatasetWithComponents(Dataset):
	def __init__(self, peak_file, genome_fa, chrom_list, mode="train", seq_len=350, rev_comp_prob=0.5, jitter=50):
		self.seq_len = seq_len
		self.peak_file = peak_file
		self.genome = pyfaidx.Fasta(genome_fa, one_based_attributes=False, sequence_always_upper=True)
		self.mode = mode
		self._rev_comp_prob = rev_comp_prob
		self._jitter = jitter
		self.chrom_list = chrom_list
		self.regions = self.load_regions()
		self.assignments, self.component_list = self.load_components()

	def __len__(self):
		return len(self.component_list)

	def load_regions(self):
		chrom_set = set(self.chrom_list)
		peak_data = pd.read_csv(self.peak_file, sep="\t", header=None)
		peak_data = peak_data.loc[peak_data[0].isin(chrom_set)].reset_index(drop=True)
		return peak_data

	def load_components(self):
		component_col = self.regions.columns[-1]
		component_list = list(np.unique(self.regions[component_col].values))
		assignments = {}
		for reg in range(len(self.regions)):
			chrom, start, end = self.regions.loc[reg, 0], self.regions.loc[reg, 1], self.regions.loc[reg, 2]
			comp = self.regions.loc[reg, component_col]
			assignments[comp] = assignments.get(comp, []) + [(chrom, start, end)]

		return assignments, component_list

	def _apply_jitter(self, start):
		jitter = random.randint(-self._jitter, self._jitter)
		start += jitter
		return start

	def __getitem__(self, idx):
		super().__init__()
		curr_component = self.component_list[idx]
		seq_to_use = np.random.randint(len(self.assignments[curr_component]))
		chrom, start, end = self.assignments[curr_component][seq_to_use]
		if self.mode == "train":
			curr_len = end - start
			start = max(0, self._apply_jitter(start))
			end = start + curr_len

		buffer = (self.seq_len - (end - start)) // 2
		dna_seq = self.genome[chrom][start-buffer:start-buffer+self.seq_len].seq
		seq = np.array(encode_sequence(dna_seq.upper()))
		if self.mode == "train" and random.random() < self._rev_comp_prob:
			seq = rev_comp(seq)
		return torch.from_numpy(seq)
