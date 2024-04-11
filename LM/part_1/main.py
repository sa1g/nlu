import copy
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from functions import eval_loop, init_weights, train_loop
from model import LM_RNN
from functools import partial


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')

    DEVICE = "cuda:0"

    ############
    # Load Data
    ############

    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    print(len(vocab))

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_batch_size = 256
    dev_batch_size = 256
    test_batch_size = 256

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=dev_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )

    vocab_len = len(lang.word2id)

    ############
    # Hyperparameters
    ############
    hid_size = 200
    emb_size = 300
    out_dropout = 0.1
    emb_dropout = 0.1
    n_layers = 1

    lr = 1
    clip = 5
    device = "cuda:0"

    n_epochs = 10
    patience = 3

    # EXPERIMENT_NAME = "lstm-do-0.1-0.1-lr-1.5-hs-300-batch-128"

    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step
    # With SGD try with an higher learning rate (> 1 for instance)

    ############
    # Model
    ############

    model = LM_RNN(
        emb_size,
        hid_size,
        vocab_len,
        pad_index=lang.word2id["<pad>"],
        out_dropout=out_dropout,
        emb_dropout=emb_dropout,
        n_layers=n_layers,
    ).to(device)
    model.apply(init_weights)

    ############
    # Optimizer and Loss
    ############

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    ############
    # Tensorboard
    ############

    EXPERIMENT_NAME = f"{model.__name__}-{optimizer.__class__.__name__}-hs-{hid_size}-es-{emb_size}-od-{out_dropout}-ed-{emb_dropout}-n_layers-{n_layers}-lr-{lr}-batch-{train_batch_size}-epochs-{n_epochs}"

    writer = SummaryWriter(log_dir=f"runs/{EXPERIMENT_NAME}")

    config = {
        "hid_size": hid_size,
        "emb_size": emb_size,
        "out_dropout": out_dropout,
        "emb_dropout": emb_dropout,
        "n_layers": n_layers,
        "lr": lr,
        "clip": clip,
        "epochs": n_epochs,
        "patience": patience,
        "train_batch_size": train_batch_size,
        "dev_batch_size": dev_batch_size,
        "test_batch_size": test_batch_size,
    }
    writer.add_text("Experiment Config", str(config))

    ############
    # Training
    ############

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
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

            writer.add_scalar("Loss/Train", losses_train[-1], epoch)
            writer.add_scalar("Loss/Test", losses_dev[-1], epoch)
            writer.add_scalar("PPL/Test", ppl_dev, epoch)

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to("cpu")
                patience = 3
            else:
                patience -= 1

            if patience <= 0:
                break

    best_model.to(device)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    writer.add_scalar("PPL/Eval", final_ppl, len(sampled_epochs))
    print("Test ppl: ", final_ppl)

    writer.close()

    ############
    # Save the model
    ############

    path = f"bin/{EXPERIMENT_NAME}.pt"
    torch.save(model.state_dict(), path)

    # # To load the model you need to initialize it
    # # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # # Then you load it
    # # model.load_state_dict(torch.load(path))
