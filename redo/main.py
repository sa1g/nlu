import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from functools import partial

from tqdm import tqdm

import logging
logging.basicConfig(format='%(msecs)s:%(levelname)s:%(message)s', level=logging.DEBUG)

from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_RNN
from functions import eval_loop, init_weights, train_loop

if __name__ == "__main__":
    DEVICE = "cuda:0"

    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_batch_size = 128
    dev_batch_size = 128
    test_batch_size = 128

    logging.debug("Dataloading init")
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
    logging.debug("Dataloading done")

    vocab_len = len(lang.word2id)

    # MODEL PARAMETERS
    emb_size = 300
    hid_size = 300
    output_size = vocab_len
    pad_index = lang.word2id["<pad>"]
    emb_dropout = 0
    out_dropout = 0
    n_layers = 1

    # HYPER PARAMETERS
    lr = 1
    n_epochs = 2
    clip = 5

    

    # MODEL SETUP
    model = LM_RNN(emb_size, hid_size, output_size, pad_index,
                   emb_dropout, out_dropout, n_layers).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"])
    criterion_eval = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum")
    logging.debug("Model done")
    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{model.name}")

    config = {
        "emb_size": emb_size,
        "hid_size": hid_size,
        "emb_dropout": emb_dropout,
        "out_dropout": out_dropout,
        "n_layers": n_layers,
        "lr": lr,
        "train_batch_size": train_batch_size,
        "dev_batch_size": dev_batch_size,
        "test_batch_size": test_batch_size
    }

    # TRAINING

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    logging.debug("Training")
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer,
                          criterion_train, model, clip)

        if epoch%1 == 0 :
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
                patience -=1

            if patience <= 0:
                break
    logging.debug("Done")
    
    best_model.to(DEVICE)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    writer.add_scalar("PPL/Eval", final_ppl, len(sampled_epochs))
    print("Test ppl: ", final_ppl)

    writer.close()

    path = f"bin/{best_model.name}.pt"
    torch.save(best_model.state_dict(), path)