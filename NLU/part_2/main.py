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
from model import IntentSlotModel
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
    train_raw, dev_raw, test_raw, slots2id, id2slots, intent2id, id2intent = (
        get_data_and_mapping()
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processed_train = preprocess_data(train_raw, tokenizer, slots2id, intent2id)

    train_dataset = ATISDataset(processed_train)
    test_dataset = ATISDataset(processed_train)
    dev_dataset = ATISDataset(processed_train)

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
        logging.debug(dataloader["intent"].shape)

        break

    name = config["name"]
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{name}")

    accuracy, f1, loss = 0, 0, [float("inf")]

    best_model = None
    best_f1 = 0

    slot_f1s, intent_acc, all_losses_train, all_losses_dev = [], [], [], []

    for _ in tqdm(range(config["runs"]), desc="Runs"):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_steps = len(train_dataloader) * config["epochs"]
        warmup_steps = int(0.1 * total_steps)

        model = IntentSlotModel(len(slots2id), len(intent2id))
        model.to(get_device())

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        intent_loss_fn = torch.nn.CrossEntropyLoss()
        slot_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=slots2id["pad"])

        # scheduler = None
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
            desc=f"Epochs | Acc: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}",
            leave=False,
        )

        for epoch in epochs_tqdm:
            t_loss = train_loop(
                model,
                train_dataloader,
                optimizer,
                intent_loss_fn,
                slot_loss_fn,
                slots2id,
                scheduler,
                config["grad_clip"],
            )

            train_losses.append(np.asarray(t_loss).mean())

            accuracy, f1, loss = eval_loop(
                model,
                dev_dataloader,
                intent_loss_fn,
                slot_loss_fn,
                tokenizer,
                id2slots,
                slots2id,
            )

            dev_losses.append(np.asarray(loss).mean())

            epochs_tqdm.set_description(
                f"Epochs | Acc: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to("cpu")

        accuracy, f1, loss = eval_loop(
            model,
            test_dataloader,
            intent_loss_fn,
            slot_loss_fn,
            tokenizer,
            id2slots,
            slots2id,
        )

        intent_acc.append(accuracy)
        slot_f1s.append(f1)

        all_losses_train.append(train_losses)
        all_losses_dev.append(dev_losses)

        print(
            f"Accuracy: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}"
        )

    avg_losses_train = np.mean(np.asarray(all_losses_train), axis=0)
    avg_losses_dev = np.mean(np.asarray(all_losses_dev), axis=0)
    std_losses_train = np.std(np.asarray(all_losses_train), axis=0)
    std_losses_dev = np.std(np.asarray(all_losses_dev), axis=0)

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    for epoch, (
        avg_loss_train,
        std_loss_train,
        avg_loss_dev,
        std_loss_dev,
    ) in enumerate(
        zip(avg_losses_train, std_losses_train, avg_losses_dev, std_losses_dev)
    ):
        writer.add_scalar("Loss/Train_avg", avg_loss_train, epoch)
        writer.add_scalar("Loss/Train_std", std_loss_train, epoch)
        writer.add_scalar("Loss/Dev_avg", avg_loss_dev, epoch)
        writer.add_scalar("Loss/Dev_std", std_loss_dev, epoch)

    writer.add_scalar("Metrics/Slot_F1_avg", slot_f1s.mean())
    writer.add_scalar("Metrics/Slot_F1_std", slot_f1s.std())
    writer.add_scalar("Metrics/Intent_Acc_avg", intent_acc.mean())
    writer.add_scalar("Metrics/Intent_Acc_std", intent_acc.std())

    print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
    print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(intent_acc.std(), 3))

    PATH = f"bin/{name}.pt"
    saving_object = {
        "model": best_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "slot2id": slots2id,
        "intent2id": intent2id,
    }
    torch.save(saving_object, PATH)

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
