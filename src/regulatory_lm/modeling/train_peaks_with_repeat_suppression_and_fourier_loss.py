import os
import sys
import time
import json
import yaml

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from .utils import save_model, print_and_log, set_seed
from .losses import FourierAttributionLoss
from .model import RegulatoryLM, InputEmbedder, SimpleLMDecoder, BicausalNet, DummyBicausalNet, DilatedConvNet
from .model import RegulatoryLM, MODULES
from ..dataloader.data_generator_peaks import NarrowpeakDatasetWithRepeatMasking, BedDatasetWithRepeatMasking
from torch.optim import Adam
from tqdm import tqdm

OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

def run_epoch(model, dloader, device, mask_prob, num_real_tokens, repeat_weight, fourier_loss_weight, motif_low, motif_high, smoothing_factor, seq_len, optimizer=None):
    '''
    -Runs an epoch of training or validation
    -mask_prob is the proportion of tokens that we use to predict (not just masked)
        these tokens are 80% masked, 10% mutated, and 10% unchanged
    -The first part of the function produces seqs_input and seqs_labels, which are the input and labels respectively
    -Our loss function ignores the index 4, which corresponds to special characters (eg. "N") in the input
        In the labels, we set all non-predicted positions to 4 as well so they are ignored by the loss function
    -We then calculate the language model and fourier loss and combine them according to the relevant parameters
    -See the example config file for descriptions of these parameters
    '''
    #If num_real_tokens are 4, then tokens are A, C, G, T, special characters, and <mask>
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=num_real_tokens, reduction="none")
    fourier_loss_fn = FourierAttributionLoss(seq_len, motif_low, motif_high, smoothing_factor, device, num_real_tokens)
    batch_losses, lm_losses, fourier_losses = [], [], []
    for i, (seqs, is_repeat) in enumerate(tqdm(dloader, ncols=100, unit="batch")):        
        true_mask_prob, corrupt_relative_prob = 0.8, 2/3 #we mask 80% of chosen tokens, and 2/3 of the remaining are randomly mutated. 
        #The 2/3 is chosen so that in expectation, 50% of the non-masked chosen tokens will be the original nucleotide (since vocab size is only 4) 
        to_predict = torch.rand(seqs.shape) < mask_prob #The overall locations we want to predict
        potential_mask_locs = torch.rand(seqs.shape) < true_mask_prob #Filter with true_mask_prob probability, when combined with true_predict will produce mask locations
        potential_corrupt_locs = torch.rand(seqs.shape) < corrupt_relative_prob #Will produce locs to corrupt when combined with prediction locations that are not masked
        to_mask = torch.logical_and(to_predict, potential_mask_locs) #Produces final list of tokens to mask
        to_corrupt = torch.logical_and(torch.logical_and(to_predict, ~potential_mask_locs), potential_corrupt_locs) #Produces final list of tokens to corrupt

        random_corruptions = torch.randint(0, 4, seqs.shape) #Random corruptions to use (4 is not inclusive)

        seqs_input = torch.where(~to_mask, seqs, num_real_tokens+1) #mask certain locations
        seqs_input = torch.where(to_corrupt, random_corruptions, seqs_input) #corrupt certain other locations
        seqs_labels = torch.where(to_predict, seqs, num_real_tokens) #Ignore everything except positions to predict (the "correct" prediction tokens are therefore implicitly included)
        if model.training:
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            assert optimizer is None
        seqs_input = seqs_input.to(device, dtype=torch.long)
        seqs_labels = seqs_labels.to(device, dtype=torch.long)
        seqs_raw = seqs.to(device, dtype=torch.long)
        is_repeat = is_repeat.to(device, dtype=torch.long)
        logits = model(seqs_input, None)
        lm_loss_unreduced = lm_loss_fn(logits.permute(0,2,1), seqs_labels)
        repeat_weight_mask = torch.where(is_repeat==1, repeat_weight, 1) #Weights repeats differently than the rest
        lm_loss = torch.mean(repeat_weight_mask * lm_loss_unreduced)
        fourier_loss = fourier_loss_fn(logits, seqs_raw)
        total_loss = lm_loss + fourier_loss_weight * fourier_loss
        batch_losses.append(total_loss.item())
        fourier_losses.append(fourier_loss.item())
        lm_losses.append(lm_loss.item())
        if model.training:
            total_loss.backward()
            optimizer.step()
    return batch_losses, fourier_losses, lm_losses


def train_model(model, train_loader, valid_loader, num_epochs, output_dir, early_stopping, patience, float_dtype, mask_prob, num_real_tokens, repeat_weight, fourier_loss_weight, motif_low, motif_high, 
smoothing_factor, seq_len,  optimizer_cls, optimizer_params):
    '''
    Performs model training. For each epoch, performs the training and validation steps and logs the relevant losses
    '''
    print("Parameter count: ", sum(p.numel() for p in model.parameters()))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device == torch.device("cuda"):
        print("Training on GPU")
        # model = model.to(device, dtype=float_dtype)
        model = model.to(device)
        print("Model Loaded")
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    if not torch.cuda.device_count() > 1:
        print("Model will be compiled")
        model = torch.compile(model)
    
    log_path = os.path.join(output_dir, "train_log.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\ttrain_loss\tvalid_loss\n")

        if early_stopping:
            counter = 0 #new ES
        best_valid_epoch_loss = float("inf")
        for epoch in range(num_epochs):
            if torch.cuda.is_available:
                torch.cuda.empty_cache()

            model.train()
            train_batch_losses, train_fourier_losses, train_lm_losses = run_epoch(model, train_loader, device, mask_prob, num_real_tokens, repeat_weight, fourier_loss_weight, motif_low, motif_high, smoothing_factor, seq_len, optimizer)
            train_epoch_loss, train_epoch_fourier_loss, train_epoch_lm_loss = np.nanmean(train_batch_losses), np.nanmean(train_fourier_losses), np.nanmean(train_lm_losses)

            with torch.no_grad():
                model.eval()
                valid_batch_losses, valid_fourier_losses, valid_lm_losses = run_epoch(model, valid_loader, device, mask_prob, num_real_tokens, repeat_weight, fourier_loss_weight, motif_low, motif_high, smoothing_factor, seq_len, optimizer=None)
            valid_epoch_loss, valid_epoch_fourier_loss, valid_epoch_lm_loss = np.nanmean(valid_batch_losses), np.nanmean(valid_fourier_losses), np.nanmean(valid_lm_losses)

            print(f"Epoch {epoch}: train_loss = {train_epoch_loss}, train_fourier_loss = {train_epoch_fourier_loss}, train_lm_loss = {train_epoch_lm_loss}, valid_loss = {valid_epoch_loss}, valid_fourier_loss = {valid_epoch_fourier_loss}, valid_lm_loss = {valid_epoch_lm_loss}")
            f.write(f"{epoch}\t{train_epoch_loss}\t{train_epoch_fourier_loss}\t{train_epoch_lm_loss}\t{valid_epoch_loss}\t{valid_epoch_fourier_loss}\t{valid_epoch_lm_loss}\n")
            f.flush()
            save_model(model, optimizer, valid_epoch_loss, epoch, os.path.join(output_dir, f"checkpoint_{epoch}.pt"))

            if valid_epoch_loss < best_valid_epoch_loss:
                best_valid_epoch_loss = valid_epoch_loss
                # save_model(model, optimizer, valid_epoch_loss, epoch, os.path.join(output_dir, "final.pt"))
                if early_stopping: #new ES
                    counter = 0 #new ES
            else: #New ES
                if early_stopping: #new ES
                    counter  += 1 #new ES
                    if counter == patience: #new ES
                        print("Stopped early") #new ES
                        break #new ES

def main():
    args_path = sys.argv[1]
    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    set_seed(args["seed"])
    # args = json.load(open(sys.argv[1], "r"))
    output_dir = args["output_dir"]
    run_dir = os.path.join(output_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    # run_dir = os.path.join(output_dir, "run_%s" % time.strftime("%Y%m%d_%H%M%S"))
    print(f"Saving to {run_dir}")

    os.makedirs(run_dir, exist_ok=True)
    args_out_path = os.path.join(run_dir, "args.json")
    with open(args_out_path, "w") as f:
        json.dump(args, f, indent=4)

    # shutil.copyfile(sys.argv[1], os.path.join(run_dir, "args.json"))

    # os.system("cp %s %sargs.json" %(sys.argv[1], run_dir))
    # txt_writer = open(run_dir + "train_log.csv", "w")

    embedder_kwargs = args.get("embedder_kwargs", {})
    encoder_kwargs = args.get("encoder_kwargs", {})
    decoder_kwargs = args.get("decoder_kwargs", {})
    model_kwargs = args.get("model_kwargs", {})

    if args["data_format"] == "narrowpeak":
        train_dataset = NarrowpeakDatasetWithRepeatMasking(args["peak_file"], args["genome"], args["train_chrom_list"], mode="train", jitter=args["max_jitter"], min_stretch=args["min_stretch"], seq_len=args["seq_len"]) #LOAD TRAIN DATALOADER USING ARGS
        valid_dataset = NarrowpeakDatasetWithRepeatMasking(args["peak_file"], args["genome"], args["valid_chrom_list"], mode="valid", jitter=args["max_jitter"], min_stretch=args["min_stretch"], seq_len=args["seq_len"]) #LOAD VALID DATALOADER USING ARGS
    elif args["data_format"] == "bed":
        train_dataset = BedDatasetWithRepeatMasking(args["peak_file"], args["genome"], args["train_chrom_list"], mode="train", jitter=args["max_jitter"], min_stretch=args["min_stretch"], seq_len=args["seq_len"]) #LOAD TRAIN DATALOADER USING ARGS
        valid_dataset = BedDatasetWithRepeatMasking(args["peak_file"], args["genome"], args["valid_chrom_list"], mode="valid", jitter=args["max_jitter"], min_stretch=args["min_stretch"], seq_len=args["seq_len"]) #LOAD VALID DATALOADER USING ARGS
        
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, pin_memory=True, num_workers=1, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, pin_memory=True, num_workers=1)

    embedder =  MODULES[args["embedder"]](args["embedding_size"],vocab_size=args["num_real_tokens"]+2, masking=True, **embedder_kwargs)
    encoder =  MODULES[args["encoder"]](args["embedding_size"], args["num_encoder_layers"], **encoder_kwargs)
    decoder = MODULES[args["decoder"]](args["embedding_size"], **decoder_kwargs)
    model = RegulatoryLM(embedder, encoder, decoder, **model_kwargs)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)


    optimizer = OPTIMIZERS[args["optimizer"]]

    float_dtype = FLOAT_DTYPES[args["float_dtype"]]
    train_model(model, train_loader, valid_loader, args["num_epochs"], run_dir, args["early_stopping"], args["patience"], float_dtype, args["mask_prob"], args["num_real_tokens"], args["repeat_weight"], args["fourier_loss_weight"], args["motif_low"], args["motif_high"], args["smoothing_factor"], args["seq_len"], optimizer, args["optimizer_params"])

if __name__ == "__main__":
    main()

