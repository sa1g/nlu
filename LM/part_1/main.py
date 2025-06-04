# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functools import partial
from typing import Optional
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim

from tqdm import tqdm
import copy
# Import everything from functions.py file
# from functions import *

from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_RNN
from functions import train_loop, eval_loop, init_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Wrtite the code to load the datasets and to run your functions
    # Print the results

    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")

    # Vocab is computed only on training set
    # We add two special tokens end of sentence and padding
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    print(f"Vocab size: {len(vocab)}")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
    )


    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step

    # With SGD try with an higher learning rate (> 1 for instance)
    # baseline 0.0001
    lr = 0.5  # This is definitely not good for SGD
    clip = 5  # Clip the gradient

    vocab_len = len(lang.word2id)

    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(
        device
    )
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )


    n_epochs = 50
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model: Optional[torch.nn.Module] = None

    pbar = tqdm(range(1, n_epochs))
    # If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to("cpu")
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    if best_model is not None:
        best_model.to(device)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test ppl: ", final_ppl)
