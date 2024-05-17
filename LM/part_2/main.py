from functions import (
    get_model,
    get_optimizer,
    init_weights,
    train,
    get_loaders_lang,
)
from torch.utils.tensorboard import SummaryWriter
import logging
import json

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c", default="config.json", help="Config file json")


def main(
    dataset_path,
    train_batch_size,
    dev_batch_size,
    test_batch_size,
    emb_size,
    hid_size,
    emb_dropout,
    out_dropout,
    n_layers,
    lr,
    n_epochs,
    clip,
    model_type,
    optim_name,
    betas,
    eps,
    weight_decay,
):
    # dataset/ptb.test.txt
    train_loader, dev_loader, test_loader, lang = get_loaders_lang(
        dataset_path, train_batch_size, dev_batch_size, test_batch_size
    )

    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]
    output_size = vocab_len

    # MODEL SETUP
    model = get_model(
        emb_size=emb_size,
        hid_size=hid_size,
        output_size=output_size,
        pad_index=pad_index,
        emb_dropout=emb_dropout,
        out_dropout=out_dropout,
        n_layers=n_layers,
        device=DEVICE,
        init_weights=init_weights,
        model_type=model_type,
    )

    optimizer = get_optimizer(
        model,
        optim_name=optim_name,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    logging.debug("Model done")

    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{model.name}")

    if optim_name == "AdamW":
        config = {
            "emb_size": emb_size,
            "hid_size": hid_size,
            "emb_dropout": emb_dropout,
            "out_dropout": out_dropout,
            "n_layers": n_layers,
            "lr": lr,
            "train_batch_size": train_batch_size,
            "dev_batch_size": dev_batch_size,
            "test_batch_size": test_batch_size,
            "n_epochs": n_epochs,
            "optim_name": optim_name,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
    else:
        config = {
            "emb_size": emb_size,
            "hid_size": hid_size,
            "emb_dropout": emb_dropout,
            "out_dropout": out_dropout,
            "n_layers": n_layers,
            "lr": lr,
            "train_batch_size": train_batch_size,
            "dev_batch_size": dev_batch_size,
            "test_batch_size": test_batch_size,
            "n_epochs": n_epochs,
        }

    # TRAINING
    train(
        model=model,
        optimizer=optimizer,
        lang=lang,
        writer=writer,
        n_epochs=n_epochs,
        clip=clip,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=DEVICE,
    )


def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs


if __name__ == "__main__":
    DEVICE = "cuda:0"

    args = parser.parse_args()
    config : dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !! ", key)
        assert config.get("model_type", "LM_RNN") in ["LM_RNN", "LM_LSTM"]
        assert config.get("optim_name", "SGD") in ["SGD", "AdamW"]

        main(
            dataset_path=config.get("dataset_path", "../dataset"),
            train_batch_size=config.get("train_batch_size", 128),
            dev_batch_size=config.get("dev_batch_size", 128),
            test_batch_size=config.get("test_batch_size", 128),
            emb_size=config.get("emb_size", 300),
            hid_size=config.get("hid_size", 300),
            emb_dropout=config.get("emb_dropout", 0),
            out_dropout=config.get("out_dropout", 0),
            n_layers=config.get("n_layers", 1),
            lr=config.get("lr", 0.0001),
            n_epochs=config.get("n_epochs", 1),
            clip=config.get("clip", 5),
            model_type=config.get("model_type", "LM_RNN"),
            optim_name=config.get("optim_name", "SGD"),
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-08),
            weight_decay=config.get("weight_decay", 0.01),
        )
