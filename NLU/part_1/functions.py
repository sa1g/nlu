# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import copy
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import Lang
from conll import evaluate
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch import optim


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


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample["utterances"], sample["slots_len"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


def train(
    model: nn.Module,
    optimizer_config: dict,
    lang: Lang,
    writer,
    n_epochs,
    clip: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    PAD_TOKEN,
    w2id,
    slot2id,
    intent2id,
    device: str = "cpu",
    patience: int = 5,
    runs: int = 5
):
    optimizer = optim.Adam(model.parameters(), lr=optimizer_config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    # Simple Training loop
    # TODO: put multiple runs to get avg and std
    # TODO: add tensorboard support
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    run_patience = patience
    best_model = None

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
                best_model = copy.deepcopy(model).to("cpu")

                run_patience = patience
            else:
                run_patience -= 1
            if run_patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )

    # Saving the model:
    # PATH = os.path.join("bin", best_model.name, "pt")
    PATH = f"bin/{best_model.name}.pt"
    saving_object = {"epoch": x,
                     "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "w2id": w2id,
                     "slot2id": slot2id,
                     "intent2id": intent2id}
    torch.save(saving_object, PATH)

    # logging.info("Slot F1: %i", results_test["total"]["f"])
    # logging.info("Intent Accuracy: %i", intent_test["accuracy"])
    print("Slot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"])

    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor("white")
    plt.title("Train and Dev Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(sampled_epochs, losses_train, label="Train loss")
    plt.plot(sampled_epochs, losses_dev, label="Dev loss")
    plt.legend()
    plt.show()
