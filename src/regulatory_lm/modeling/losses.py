import torch
from torch import nn
import math
import numpy as np
import scipy

class FourierAttributionLoss(torch.nn.Module):
    def __init__(self, seq_len, motif_low_len, motif_high_len, smoothing_factor, device, num_real_tokens):
        '''
        Uses a fourier-transform based loss function to lower the loss weights on features of length above and below a certain window
        '''
        super().__init__()
        self.seq_len = seq_len
        self.motif_low_len = motif_low_len
        self.motif_high_len = motif_high_len
        self.smoothing_factor = smoothing_factor
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.weights = self.calc_weights()
        self.num_real_tokens = num_real_tokens

    def calc_weights(self):
        '''
        Here, we calculate the weight for every frequency value returned by fft
        We have bounds for expected motif lengths, and we set weights to 1 within the bounds
        Weights outside the bounds are set according to the formulas below
        '''
        high_freq_cutoff = self.seq_len // self.motif_low_len
        low_freq_cutoff = self.seq_len // self.motif_high_len + 1
        weights = []
        for pos in range(1, self.seq_len // 2 + 1):
            if pos < low_freq_cutoff:
                curr_weight = 1 / (1 + (low_freq_cutoff - pos)**self.smoothing_factor)
            elif pos <= high_freq_cutoff: 
                curr_weight = 1
            else:
                curr_weight = 1 / (1 + (pos - high_freq_cutoff)**self.smoothing_factor)
                
            weights.append(curr_weight)

        return torch.tensor(weights).to(self.device)

    def forward(self, logits, seqs):
        '''
        Loss calculation occurs by the following steps:
        1. Calculate per-base probabilities from the logits and only keep the probabilities for the true bases
        2. Apply FFT and remove the DC component, take magnitude of each value
        3. Divide by the L1 norm and apply the weights (ensuring each row sums to at most 1)
        4. Sum, subtract from 1, and finally take the mean
        '''
        probs = self.softmax(logits)
        n_mask = (seqs >= 0) & (seqs < self.num_real_tokens) #To handle N cases
        probs = probs.gather(dim=-1, index=seqs.clamp(0, self.num_real_tokens-1).unsqueeze(-1)).squeeze(-1) * n_mask
        fft_vals = torch.abs(torch.fft.rfft(probs))[:,1:]
        fft_vals = fft_vals / (1e-8 + torch.norm(fft_vals, p=1, dim=-1)).unsqueeze(-1)
        weighted_fft_vals = fft_vals * self.weights
        samplewise_fft_loss = 1 - weighted_fft_vals.sum(-1)
        return samplewise_fft_loss.mean()


class FourierAttributionLossSmoothed(torch.nn.Module):
    def __init__(self, seq_len, motif_low_len, motif_high_len, smoothing_factor, device, num_real_tokens, smooth_sigma=3):
        '''
        Uses a fourier-transform based loss function to lower the loss weights on features of length above and below a certain window
        '''
        super().__init__()
        self.seq_len = seq_len
        self.motif_low_len = motif_low_len
        self.motif_high_len = motif_high_len
        self.smoothing_factor = smoothing_factor
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.weights = self.calc_weights()
        self.num_real_tokens = num_real_tokens
        self.smooth_sigma = smooth_sigma
        self.kernel, self.final_sigma = self.create_smoothing_kernel()
        
    def create_smoothing_kernel(self):
        # Generate the kernel
        if self.smooth_sigma == 0:
            sigma, truncate = 1, 0
        else:
            sigma, truncate = self.smooth_sigma, 1
        base = np.zeros(1 + (2 * sigma))
        base[sigma] = 1  # Center of window is 1 everywhere else is 0
        kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
        kernel = torch.tensor(kernel, device=self.device)
        return kernel, sigma
       
        
    def smooth_tensor_1d(self, input_tensor):
        """
        Smooths an input tensor along a dimension using a Gaussian filter.
        Arguments:
            `input_tensor`: a A x B tensor to smooth along the second dimension
            `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
                standard deviation of the Gaussian to use, and the Gaussian will be
                truncated after 1 sigma (i.e. the smoothing window is
                1 + (2 * sigma); sigma of 0 means no smoothing
        Returns an array the same shape as the input tensor, with the dimension of
        `B` smoothed.
        """

        # Expand the input and kernel to 3D, with channels of 1
        # Also make the kernel float-type, as the input is going to be of type float
        input_tensor = torch.unsqueeze(input_tensor, dim=1)
        kernel = torch.unsqueeze(torch.unsqueeze(self.kernel, dim=0), dim=1).float()

        smoothed = torch.nn.functional.conv1d(
            input_tensor, kernel, padding=self.final_sigma
        )

        return torch.squeeze(smoothed, dim=1)


    def calc_weights(self):
        '''
        Here, we calculate the weight for every frequency value returned by fft
        We have bounds for expected motif lengths, and we set weights to 1 within the bounds
        Weights outside the bounds are set according to the formulas below
        '''
        high_freq_cutoff = self.seq_len // self.motif_low_len
        low_freq_cutoff = self.seq_len // self.motif_high_len + 1
        weights = []
        for pos in range(1, self.seq_len // 2 + 1):
            if pos < low_freq_cutoff:
                curr_weight = 1 / (1 + (low_freq_cutoff - pos)**self.smoothing_factor)
            elif pos <= high_freq_cutoff: 
                curr_weight = 1
            else:
                curr_weight = 1 / (1 + (pos - high_freq_cutoff)**self.smoothing_factor)
                
            weights.append(curr_weight)

        return torch.tensor(weights).to(self.device)

    def forward(self, logits, seqs):
        '''
        Loss calculation occurs by the following steps:
        1. Calculate per-base probabilities from the logits and only keep the probabilities for the true bases
        2. Apply FFT and remove the DC component, take magnitude of each value
        3. Divide by the L1 norm and apply the weights (ensuring each row sums to at most 1)
        4. Sum, subtract from 1, and finally take the mean
        '''
        probs = self.softmax(logits)
        n_mask = (seqs >= 0) & (seqs < self.num_real_tokens) #To handle N cases
        probs = probs.gather(dim=-1, index=seqs.clamp(0, self.num_real_tokens-1).unsqueeze(-1)).squeeze(-1) * n_mask
        probs = self.smooth_tensor_1d(probs)
        fft_vals = torch.abs(torch.fft.rfft(probs))[:,1:]
        fft_vals = fft_vals / (1e-8 + torch.norm(fft_vals, p=1, dim=-1)).unsqueeze(-1)
        weighted_fft_vals = fft_vals * self.weights
        samplewise_fft_loss = 1 - weighted_fft_vals.sum(-1)
        return samplewise_fft_loss.mean()


class HighLikelihoodLoss(torch.nn.Module):
    def __init__(self, window_size, stride, avg_prob_cutoff, num_real_tokens):
        '''
        Penalizes the presence of long windows of high probability
        '''
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.avg_prob_cutoff = avg_prob_cutoff
        self.softmax = nn.Softmax(dim=-1)
        self.num_real_tokens = num_real_tokens

    def forward(self, logits, seqs):
        '''
        Calculates the sums of likelihoods in specified windows along the sequence
        Applies a penalty if the sum is 
        '''
        probs = self.softmax(logits)
        n_mask = (seqs >= 0) & (seqs < self.num_real_tokens) #To handle N cases
        probs = probs.gather(dim=-1, index=seqs.clamp(0, self.num_real_tokens-1).unsqueeze(-1)).squeeze(-1) * n_mask
        probs_window_mean = probs.unfold(-1, self.window_size, self.stride).mean(-1)
        penalty_vals = torch.clamp(probs_window_mean - self.avg_prob_cutoff, min=0)
        return penalty_vals.mean()

