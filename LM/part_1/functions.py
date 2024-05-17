import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import logging

from tqdm import tqdm

from utils import read_file, Lang, PennTreeBank, collate_fn
from model import LM_RNN, LM_LSTM


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion, model: torch.nn.Module, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def get_loaders_lang(
    path: str,
    train_batch_size: int = 128,
    dev_batch_size: int = 128,
    test_batch_size: int = 128,
) -> list[DataLoader | Lang]:
    logging.debug("Dataloading init")

    train_raw = read_file(f"{path}/ptb.train.txt")
    dev_raw = read_file(f"{path}/ptb.valid.txt")
    test_raw = read_file(f"{path}/ptb.test.txt")

    # vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

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

    logging.debug("Dataloading done")
    return train_loader, dev_loader, test_loader, lang


def get_model(
    emb_size: int,
    hid_size: int,
    output_size: int,
    pad_index,
    emb_dropout: float,
    out_dropout: float,
    n_layers: int,
    device: float = "cpu",
    init_weights=False,
    model_type: str = "LM_RNN",
) -> nn.Module:
    if model_type == "LM_RNN":
        model = LM_RNN(
            emb_size,
            hid_size,
            output_size,
            pad_index,
            emb_dropout,
            out_dropout,
            n_layers,
        ).to(device)
    elif model_type == "LM_LSTM":
        model = LM_LSTM(
            emb_size,
            hid_size,
            output_size,
            pad_index,
            emb_dropout,
            out_dropout,
            n_layers,
        ).to(device)

    if init_weights:
        model.apply(init_weights)

    return model


def get_optimizer(model: nn.Module, optim_name: str = "SGD", lr: float = 0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
    if optim_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif optim_name == "AdamW":
        return optim.AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )


def train(
    model: nn.Module,
    optimizer,
    lang: Lang,
    writer,
    n_epochs,
    clip: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
):
    """
    TODO: add docs
    Talk about early stopping, PPL and evaluation
    """
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    logging.debug("Training")
    pbar = tqdm(range(1, n_epochs))

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
    logging.debug("Done")

    best_model.to(device)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    writer.add_scalar("PPL/Eval", final_ppl, len(sampled_epochs))
    logging.info("Test ppl: %f", final_ppl)

    path = f"bin/{best_model.name}.pt"
    torch.save(best_model.state_dict(), path)
