# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from pprint import pprint
from model import ModelIAS
from utils import load_data, get_datasets
from functions import *
import os
import random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

# Global variables
device = "cuda:0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
PAD_TOKEN = 0

if __name__ == "__main__":
    # !wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
    # !wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json
    # !wget https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py

    tmp_train_raw = load_data(os.path.join("..", "dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("..", "dataset", "ATIS", "test.json"))

    train_loader, dev_loader, test_loader, lang = get_datasets(os.path.join("..", "dataset", "ATIS", "train.json"), os.path.join("..", "dataset", "ATIS", "test.json"))

    # Training setup
    hid_size = 200
    emb_size = 300

    lr = 0.0001  # learning rate
    clip = 5  # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(
        hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    # Simple Training loop
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1, n_epochs)):
        loss = train_loop(
            train_loader,
            optimizer,
            criterion_slots,
            criterion_intents,
            model,
            clip=clip,
        )
        if x % 5 == 0:  # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev["total"]["f"]
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )
    print("Slot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"])

    # Saving the model:
    # PATH = os.path.join("bin", model_name)
    # saving_object = {"epoch": x,
    #                  "model": model.state_dict(),
    #                  "optimizer": optimizer.state_dict(),
    #                  "w2id": w2id,
    #                  "slot2id": slot2id,
    #                  "intent2id": intent2id}
    # torch.save(saving_object, PATH)

    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor("white")
    plt.title("Train and Dev Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(sampled_epochs, losses_train, label="Train loss")
    plt.plot(sampled_epochs, losses_dev, label="Dev loss")
    plt.legend()
    plt.show()

    # Multiple runs
    # To have reliable results on small corpora we have to train and test the model
    # from scratch for several times. At the end, we average the results and we
    # compute the standard deviation.

    # hid_size = 200
    # emb_size = 300

    # lr = 0.0001 # learning rate
    # clip = 5 # Clip the gradient

    # out_slot = len(lang.slot2id)
    # out_int = len(lang.intent2id)
    # vocab_len = len(lang.word2id)

    # n_epochs = 200
    # runs = 5

    # slot_f1s, intent_acc = [], []
    # for x in tqdm(range(0, runs)):
    #     model = ModelIAS(hid_size, out_slot, out_int, emb_size,
    #                     vocab_len, pad_index=PAD_TOKEN).to(device)
    #     model.apply(init_weights)

    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    #     criterion_intents = nn.CrossEntropyLoss()

    #     patience = 3
    #     losses_train = []
    #     losses_dev = []
    #     sampled_epochs = []
    #     best_f1 = 0
    #     for x in range(1,n_epochs):
    #         loss = train_loop(train_loader, optimizer, criterion_slots,
    #                         criterion_intents, model)
    #         if x % 5 == 0:
    #             sampled_epochs.append(x)
    #             losses_train.append(np.asarray(loss).mean())
    #             results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
    #                                                         criterion_intents, model, lang)
    #             losses_dev.append(np.asarray(loss_dev).mean())
    #             f1 = results_dev['total']['f']

    #             if f1 > best_f1:
    #                 best_f1 = f1
    #             else:
    #                 patience -= 1
    #             if patience <= 0: # Early stopping with patient
    #                 break # Not nice but it keeps the code clean

    #     results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
    #                                             criterion_intents, model, lang)
    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
