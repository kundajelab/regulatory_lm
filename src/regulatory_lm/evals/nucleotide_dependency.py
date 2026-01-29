import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#L*4*L
#First L - Which nucleotide is changed
#4 - what it's changed to
#Second L - sequence dimension

device = "cuda" if torch.cuda.is_available() else "cpu"

def produce_seqs_with_all_mutations(input_seq):
    '''
    Takes in a DNA sequence with length L and tokens 0,1,2,3
    Produces a L x 4 x L tensor, where index n is a 4xL tensor with all four possible nucleotides at position n and all other positions equal to the original sequence
    The slice of the tensor [n,:,n] will equal [0,1,2,3]
    '''
    #input_seq is a length-L tensor
    input_len = len(input_seq)
    #Create tensor with output shape by repeating input
    input_repeat = input_seq.repeat(input_len * 4).reshape([input_len, 4, input_len])
    #Substitute [0,1,2,3] at each position where first and third dimensions match
    input_repeat[list(range(input_len)), :, list(range(input_len))] = torch.tensor([0,1,2,3])
    return input_repeat

def predict_all_seqs(seq_input, cat, model):
    '''
    Takes in a tensor of shape Lx4xL as produced in the previous function, gets model predictions from them, and applies softmax
    Returns a probability tensor of shape Lx4xLx4 (corresponds to predictions for all 4 nucleotides at each token in the input)
    '''
    model = model.to(device)
    if type(cat) == int:
        cat_tensor = torch.tensor([cat] * len(seq_input))
    elif type(cat) == torch.Tensor:
        cat_tensor = cat.repeat((len(seq_input), 1))
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    batch_preds = []
    #We create a dataloader on the fly to produce our data. We will predict on 64 positions at a time
    dataloader = data.DataLoader(data.TensorDataset(seq_input, cat_tensor), batch_size=64)
    for i, (seqs, cats) in enumerate(tqdm(dataloader, ncols=100, unit="batch")):
        with torch.no_grad():
            seqs = seqs.to(device)
            seqs_shape = seqs.shape
            if type(cat) == int:
                cats = torch.repeat_interleave(cats.to(device), 4)
            elif type(cat) == torch.Tensor:
                cats = cats.repeat((4,1))
            #We make predictions, needing to reshape seqs to 2-D to comply with the required model inputs
            direct_model_preds = model(seqs.reshape([seqs_shape[0]*seqs_shape[1], seqs_shape[2]]), cats)
            #We take the softmax to convert logits to probabilities
            raw_preds = softmax(direct_model_preds[0]) if type(direct_model_preds) == tuple else softmax(direct_model_preds)
            #We finally reshape the outputs to the shape Lx4xLx4
            batch_preds.append(raw_preds.reshape([seqs_shape[0], seqs_shape[1], raw_preds.shape[1], raw_preds.shape[2]]))
    return torch.cat(batch_preds)

def calc_scores(pred_probs, input_seq, epsilon):
    '''
    Given probabilities from the previous function, calculates nucleotide dependency scores
    Produces an LxL matrix
    '''
    #Adds epsilon to predicted probs and then re-normalizes them
    pred_probs += epsilon
    pred_probs = pred_probs / pred_probs.sum(-1).unsqueeze(-1)
    #Creates index_mask, which tells us which nucleotides actually comprised the true sequence
    index_mask = torch.zeros(pred_probs.shape)
    index_mask[torch.arange(pred_probs.size()[0]),input_seq,:,:] = 1
    #We use the index_mask to separate the ground truth preds from the mutation preds
    ground_truth_preds = pred_probs[index_mask==1].reshape([len(input_seq), 1, len(input_seq), 4])
    mutate_preds = pred_probs[index_mask==0].reshape([len(input_seq), 3, len(input_seq), 4])
    #We convert the probabilities to odds
    ground_truth_odds = torch.div(ground_truth_preds, 1-ground_truth_preds) 
    mutate_odds = torch.div(mutate_preds, 1-mutate_preds)
    #The scores are then simply the max of the log ratio of the odds
    log_odds_diff = torch.abs(torch.log2(mutate_odds) - torch.log2(ground_truth_odds))
    diff_maxes = log_odds_diff.max(-1)[0].max(1)[0] #torch max is weird
    diff_maxes = diff_maxes.fill_diagonal_(0.0)
    return diff_maxes

def plot_heatmap(scores, input_seq, out_file=None):
    #Input is NUMPY ARRAYS
    reverse_one_hot = {0:"A", 1:"C", 2:"G", 3:"T", 4:"N"}
    dna_seq= [reverse_one_hot[int(x)] for x in input_seq]
    plt.figure(dpi=300)
    sns.heatmap(scores.cpu().numpy(), xticklabels=dna_seq, yticklabels=dna_seq)
    if out_file is not None:
        plt.savefig(out_file, dpi=300)
    plt.title("Nucleotide Dependencies")
    plt.show()
    

def run_dependency_pipeline(model, input_seq, cat, out_file=None, epsilon=1e-10):
    seqs_with_mutations = produce_seqs_with_all_mutations(input_seq)
    predictions = predict_all_seqs(seqs_with_mutations, cat, model)
    scores = calc_scores(predictions, input_seq, epsilon)
    plot_heatmap(scores, input_seq, out_file)
    return scores