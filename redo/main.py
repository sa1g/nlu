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

    optimizer = get_optimizer(model, optim_name="SGD", lr=lr)

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
        "test_batch_size": test_batch_size,
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
    config = load_config(args.c)

    for config in config.values():
        assert config["model_type"] in ["LM_RNN"]

        if config["model_type"] == "LM_RNN":
            main(
                dataset_path=config["dataset_path"],
                train_batch_size=config["train_batch_size"],
                dev_batch_size=config["dev_batch_size"],
                test_batch_size=config["test_batch_size"],
                emb_size=config["emb_size"],
                hid_size=config["hid_size"],
                emb_dropout=config["emb_dropout"],
                out_dropout=config["out_dropout"],
                n_layers=config["n_layers"],
                lr=config["lr"],
                n_epochs=config["n_epochs"],
                clip=config["clip"],
                model_type=config["model_type"],
            )
