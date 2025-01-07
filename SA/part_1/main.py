import copy
import json
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


from utils import (
    get_data_and_mapping,
    preprocess_data,
    ATISDataset,
    collate_fn,
    get_device,
)
from model import SlotModel
from functions import train_loop, eval_loop

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", default="config.json", help="Config file json")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.CRITICAL)


def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs


def main(config: dict):
    train_raw, dev_raw, test_raw, slots2id, id2slots = get_data_and_mapping()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processed_train = preprocess_data(train_raw, tokenizer, slots2id)
    processed_test = preprocess_data(test_raw, tokenizer, slots2id)
    processed_dev = preprocess_data(dev_raw, tokenizer, slots2id)

    train_dataset = ATISDataset(processed_train)
    test_dataset = ATISDataset(processed_test)
    dev_dataset = ATISDataset(processed_dev)

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=config["dev_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    for dataloader in dev_dataloader:
        logging.debug(dataloader["input_ids"].shape)
        logging.debug(dataloader["attention_mask"].shape)
        logging.debug(dataloader["token_type_ids"].shape)
        logging.debug(dataloader["slots"].shape)

        logging.debug(dataloader["input_ids"][0])
        logging.debug(dataloader["attention_mask"][0])
        logging.debug(dataloader["token_type_ids"][0])
        logging.debug(dataloader["slots"][0])
        break

    name = config["name"]
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{name}")

    f1, precision, recall, loss = (
        -float("inf"),
        -float("inf"),
        -float("inf"),
        [float("inf")],
    )

    best_model = None
    best_f1 = -float("inf")

    rolling_f1, rolling_prec, rolling_recall, rolling_loss = [], [], [], []

    for r in tqdm(range(config["runs"]), desc="Runs"):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_steps = len(train_dataloader) * config["epochs"]
        warmup_steps = int(0.1 * total_steps)

        model = SlotModel(len(slots2id))
        model.to(get_device())

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        slot_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=slots2id["pad"])

        if config["scheduler"]:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(train_dataloader) * config["epochs"],
            )
        else:
            scheduler = None

        train_losses = []
        dev_losses = []

        epochs_tqdm = tqdm(
            range(config["epochs"]),
            desc=f"Epochs | f1: {f1} - precision: {precision} - recall: {recall} - Loss: {sum(loss)/len(loss):.4f}",
            leave=False,
        )

        for epoch in epochs_tqdm:
            t_loss = train_loop(
                model,
                train_dataloader,
                optimizer,
                slot_loss_fn,
                slots2id,
                scheduler,
                config["grad_clip"],
            )

            train_losses.append(np.asarray(t_loss).mean())

            f1, precision, recall, loss = eval_loop(
                model, dev_dataloader, slot_loss_fn, tokenizer, id2slots, slots2id
            )

            dev_losses.append(np.asarray(loss).mean())

            epochs_tqdm.set_description(
                f"Epochs | f1: {f1} - precision: {precision} - recall: {recall} - Loss: {sum(loss)/len(loss):.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to("cpu")
            # continue 160

            writer.add_scalar("Loss/train", train_losses[-1], epoch)
            writer.add_scalar("Loss/dev", dev_losses[-1], epoch)
            writer.add_scalar("F1/dev", f1, epoch)
            writer.add_scalar("Precision/dev", precision, epoch)
            writer.add_scalar("Recall/dev", recall, epoch)

        f1, precision, recall, loss = eval_loop(
            model, test_dataloader, slot_loss_fn, tokenizer, id2slots, slots2id
        )

        writer.add_scalar("F1/test", f1, r)
        writer.add_scalar("Precision/test", precision, r)
        writer.add_scalar("Recall/test", recall, r)

        rolling_f1.append(f1)
        rolling_prec.append(precision)
        rolling_recall.append(recall)
        # rolling_loss.append(loss)

    writer.add_scalar("F1/test/mean", np.mean(rolling_f1), 0)
    writer.add_scalar("Precision/test/mean", np.mean(rolling_prec), 0)
    writer.add_scalar("Recall/test/mean", np.mean(rolling_recall), 0)

    PATH = f"bin/{name}.pt"
    saving_object = {
        "model": best_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "slot2id": slots2id,
    }
    torch.save(saving_object, PATH)
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    config: dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !!", key)

        config = {
            "name": key,
            "dataset_path": config.get(
                "dataset_path", os.path.join("..", "dataset", "ATIS")
            ),
            "train_batch_size": config.get("train_batch_size", 64),
            "dev_batch_size": config.get("dev_batch_size", 64),
            "test_batch_size": config.get("test_batch_size", 64),
            "runs": config.get("runs", 5),
            "epochs": config.get("epochs", 6),
            "grad_clip": config.get("grad_clip", True),
            "scheduler": config.get("scheduler", True),
        }

        main(config)
