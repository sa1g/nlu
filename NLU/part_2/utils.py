# Add functions or classes used for data loading and preprocessing


import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional

from transformers import BertTokenizerFast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = 0

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@dataclass(frozen=True)
class Common:
    """
    common configuration for the experiments
    """

    dataset_base_path: str = "../dataset/ATIS/"
    train_batch_size: int = 128
    eval_batch_size: int = 64
    test_batch_size: int = 64


@dataclass
class ExperimentConfig:
    """
    Configuration for experiments
    """

    name: str = "Baseline"
    hid_size: int = 200
    emb_size: int = 300
    n_layer: int = 1
    lr: float = 0.0001
    clip: int = 5
    n_epochs: int = 200
    n_runs: int = 5
    patience: int = 3
    log_inner: bool = True
    bidirectional: bool = False
    emb_dropout: float = 0.0
    out_dropout: float = 0.0

    optim: type[torch.optim.Adam] = torch.optim.Adam


def load_data(path):
    """
    input: path/to/data
    output: json
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang:
    # def __init__(self, words, intents, slots, cutoff=0):
    def __init__(self, intents, slots):
        # transformers.models.bert.tokenization_bert.BertTokenizer
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        self.pad_token = self.tokenizer.pad_token_id
        self.pad_label = PAD_TOKEN

        # self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)

        # self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


@dataclass
class Sample:
    utterance: torch.Tensor
    attention_mask: torch.Tensor
    slots: torch.Tensor
    intent: torch.Tensor


class IntentsAndSlots(Dataset):
    def __init__(self, dataset, lang: Lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = "unk"
        self.tokenizer = lang.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        self.processed_samples = [
            self.preprocess(self.utterances[i], self.slot_ids[i], self.intent_ids[i], i)
            for i in range(len(self.utterances))
        ]

    def preprocess(self, phrase, slot_ids, intent_ids, i):
        """
        Prendo una frase, i suoi slot e il suo intent.

        Quando la frase viene tokenizzata, e' possibile che ci siano subtokens. Quindi
        devo mappare gli slot cosi' che quando ho subtoken, il primo subtoken abbia l'id corretto,
        mentre i successivi siano padding. Cosi' non sminchiano il significato degli slot,
        ne l'accuratezza e non vengono calcolati nella loss, ne' nella valutazione.
        """

        tokenized_sentence = []
        attention_mask = []
        slots = []

        for word, label in zip(phrase.split(), slot_ids):
            tokenized = self.tokenizer(word, add_special_tokens=False)

            utt = tokenized["input_ids"]  # type: ignore
            att_mask = tokenized["attention_mask"]  # type: ignore
            word_ids = tokenized.word_ids()

            tokenized_sentence.extend(utt)
            attention_mask.extend(att_mask)

            # If all word_ids are the same, it means the word is a single token
            # so we have to <pad> the slot_ids which are not the first one
            if len(set(word_ids)) == 1:
                slots.extend([label])
                slots.extend([self.pad_token_id] * (len(utt) - 1))
            else:
                # otherwise, we can copy the slot_id for all the tokens of the "word"
                slots.extend([label] * len(utt))

        tokenized_sentence = torch.tensor(tokenized_sentence)
        attention_mask = torch.tensor(attention_mask)
        slots = torch.tensor(slots)
        intent_ids = torch.tensor(intent_ids)

        return Sample(
            utterance=tokenized_sentence,
            attention_mask=attention_mask,
            slots=slots,
            intent=intent_ids,
        )

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx) -> Sample:
        return self.processed_samples[idx]

    # Auxiliary methods
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data, device: torch.device):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["utterance"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item["utterance"])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


def get_dataloaders_and_lang(
    config: Common,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, Lang]:
    tmp_train_raw = load_data(os.path.join(config.dataset_base_path, "train.json"))
    test_raw = load_data(os.path.join(config.dataset_base_path, "test.json"))

    logging.debug(f"Train samples: {len(tmp_train_raw)}")
    logging.debug(f"Test samples: {len(test_raw)}")
    logging.debug(f"Train samples: {tmp_train_raw[0]}")

    portion = 0.10

    intents = [x["intent"] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    # Dataset size
    logging.info(f"TRAIN size: {len(train_raw)}")
    logging.info(f"DEV size: {len(dev_raw)}")
    logging.info(f"TEST size: {len(test_raw)}")

    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(intents=intents, slots=slots)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # # Dataloader instantiations
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.train_batch_size,
    #     collate_fn=lambda x: collate_fn(x, device=device),
    #     shuffle=True,
    # )
    # dev_loader = DataLoader(
    #     dev_dataset,
    #     batch_size=config.eval_batch_size,
    #     collate_fn=lambda x: collate_fn(x, device=device),
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=config.test_batch_size,
    #     collate_fn=lambda x: collate_fn(x, device=device),
    # )

    return train_loader, dev_loader, test_loader, lang


class EmojiFormatter(logging.Formatter):
    def format(self, record):
        # Add emoji based on log level
        if record.levelno == logging.INFO:
            emoji = "🤓\t"
        elif record.levelno == logging.WARNING:
            emoji = "⚠️\t"
        elif record.levelno == logging.DEBUG:
            emoji = "❗\t"
        elif record.levelno == logging.ERROR:
            emoji = "❌\t"
        elif record.levelno == logging.CRITICAL:
            emoji = "🚨\t"
        else:
            emoji = ""  # Default (no emoji for other levels)

        # Update the format string dynamically
        self._style._fmt = f"{emoji}%(levelname)s - %(asctime)s - %(message)s"
        return super().format(record)


# Replace the default formatter with our custom one
formatter = EmojiFormatter(datefmt="%Y-%m-%d %H:%M:%S")  # Optional: Customize timestamp
for handler in logger.handlers:
    handler.setFormatter(formatter)
