import numpy as np

MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
def encode_sequence(sequence): 
    encoded_sequence = [MAPPING.get(nucleotide, 4) for nucleotide in sequence]
    return encoded_sequence

# def rev_comp_sequence(sequence):
#         complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'R': 'Y', 'Y': 'R', 'M': 'K', 'K': 'M', 'S': 'S', 'W': 'W', 'N': 'N'}
#         return [complement[base] for base in reversed(sequence)]

def rev_comp(seq):
    return (3 - seq[::-1]) % 5