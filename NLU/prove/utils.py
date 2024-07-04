from collections import Counter
import json
import os

import logging

import torch
from transformers import BertTokenizer

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset


def load_data(path):
    dataset = []
    with open(path, "r") as file:
        dataset = json.loads(file.read())

    return dataset


def split_sets(tmp_train_raw):
    portion = 0.10

    intents = [x["intent"] for x in tmp_train_raw]  # We stratify on intents
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

    return train_raw, dev_raw


def get_data(
    train=os.path.join("../dataset", "ATIS", "train.json"),
    test=os.path.join("../dataset", "ATIS", "test.json"),
):

    tmp_train_raw = load_data(train)
    test_raw = load_data(test)

    train_raw, dev_raw = split_sets(tmp_train_raw)

    logging.info("Train size: %d", len(train_raw))
    logging.info("Dev size: %d", len(dev_raw))
    logging.info("Test size: %d", len(test_raw))

    return train_raw, dev_raw, test_raw


class Tokenizer:
    def __init__(self, train_raw, dev_raw, test_raw):
        """
        1. Get unique
            - slots
            - intents
        2. Add them to the tokenizer
        3. Add special tokens to the remap
        3. Remap the data
        4. Expose mapping functions
        5. expose data
        """

        """
        GET UNIQUE SLOTS AND INTENTS
        """
        slots_set = set()
        intents_set = set()

        for phrases in [train_raw, dev_raw, test_raw]:
            for phrase in phrases:
                for slot in phrase["slots"].split():
                    slots_set.add(slot)
                intents_set.add(phrase["intent"])

        """
        ADD NEW TOKENS TO THE TOKENIZER
        """
        # as suggested here: https://stackoverflow.com/questions/62082938/how-to-stop-bert-from-breaking-apart-specific-words-into-word-piece
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        logging.debug(
            "Before adding new tokens to the tokenizer: %i",
            len(self.tokenizer.get_vocab()),
        )

        for slot in slots_set:
            tokenized = self.tokenizer.encode(slot, add_special_tokens=False)
            if len(tokenized) > 1:
                self.tokenizer.add_tokens([slot])

        for intention in intents_set:
            tokenized = self.tokenizer.encode(intention, add_special_tokens=False)
            if len(tokenized) > 1:
                self.tokenizer.add_tokens([intention])

        """ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA """

        slot_seq_id_2_bert_id = {0: 0}
        for slot in slots_set:
            slot_seq_id_2_bert_id[len(slot_seq_id_2_bert_id)] = self.tokenizer.encode(
                slot, add_special_tokens=False
            )[0]

        intent_seq_id_2_bert_id = {}
        for intention in intents_set:
            intent_seq_id_2_bert_id[len(intent_seq_id_2_bert_id)] = (
                self.tokenizer.encode(intention, add_special_tokens=False)[0]
            )

        self.slot_seq_id_2_bert_id = slot_seq_id_2_bert_id
        self.intent_seq_id_2_bert_id = intent_seq_id_2_bert_id

        self.bert_id_2_slot_seq_id = {v: k for k, v in slot_seq_id_2_bert_id.items()}
        self.bert_id_2_intent_seq_id = {
            v: k for k, v in intent_seq_id_2_bert_id.items()
        }

        self.slot_len = len(self.slot_seq_id_2_bert_id)
        self.intent_len = len(self.intent_seq_id_2_bert_id)

        logging.info("Unique slots: %d", self.slot_len)
        logging.info("Unique intents: %d", self.intent_len)

        """ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA """

        logging.debug(
            "After adding new tokens to the tokenizer: %i",
            len(self.tokenizer.get_vocab()),
        )

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            phrase = "I-fare_basis_code O O hello airfare+flight_time ground_service"
            tokenized = self.tokenizer.encode(phrase, add_special_tokens=False)
            logging.debug("Test phrase: %s", phrase)
            logging.debug("Encoded phrase: %s", tokenized)
            logging.debug("Decoded phrase: %s", self.tokenizer.decode(tokenized))

    # def __call__(self, *args, **kwds):
    #     return self.tokenizer(*args, **kwds)

    # def __getattr__(self, name):
    #     """
    #     Delegate attribute access to self.tokenizer.
    #     """
    #     return getattr(self.tokenizer, name)


class MyDataset(Dataset):
    def __init__(self, data: list, tokenizer: Tokenizer, device: str = "cuda:0"):
        self.tokenizer = tokenizer
        self.utterance = [d["utterance"] for d in data]
        self.slots = [d["slots"] for d in data]
        self.intent = [d["intent"] for d in data]

        self.device = device

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, index):
        inputs = self.utterance[index]
        slots = self.slots[index]
        intent = self.intent[index]

        # Tokenize inputs
        tmp_encoded_inputs = self.tokenizer.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
        )

        encoded_input_ids = tmp_encoded_inputs["input_ids"]
        encoded_attention_mask = tmp_encoded_inputs["attention_mask"]

        tmp_encoded_slots = self.tokenizer.tokenizer(
            slots,
            # return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
            add_special_tokens=False,
        )

        encoded_slots = tmp_encoded_slots["input_ids"]

        tmp_encoded_intent = self.tokenizer.tokenizer(
            intent,
            # return_tensors="pt",
            add_special_tokens=False,
        )

        encoded_intent = tmp_encoded_intent["input_ids"]

        encoded_input_ids = encoded_input_ids.squeeze(0).to(self.device)
        encoded_attention_mask = encoded_attention_mask.squeeze(0).to(self.device)

        # Keeping the same shape as encoded_intent remap valus with bert_id_2_intent_seq_id
        encoded_intent = torch.tensor([self.tokenizer.bert_id_2_intent_seq_id[e] for e in encoded_intent])
        encoded_slots = torch.tensor([self.tokenizer.bert_id_2_slot_seq_id[e] for e in encoded_slots])

        # encoded_intent = encoded_intent.squeeze(1).to(self.device)
        # encoded_slots = encoded_slots.squeeze(0).to(self.device)
        encoded_intent = encoded_intent.to(self.device)
        encoded_slots = encoded_slots.to(self.device)


        # print(f"input_ids: {encoded_input_ids.shape}")
        # print(f"attention_mask: {encoded_attention_mask.shape}")
        # print(f"intent_labels: {encoded_intent.shape}")
        # print(f"slot_labels: {encoded_slots.shape}")

        # exit()
        # input_ids: torch.Size([64])
        # attention_mask: torch.Size([64])
        # intent_labels: torch.Size([1])
        # slot_labels: torch.Size([64])

        return encoded_input_ids, encoded_attention_mask, encoded_intent, encoded_slots


def create_dataset(data: list, tokenizer: Tokenizer, device: str = "cpu"):

    dataset = MyDataset(data, tokenizer, device)

    # TODO: maybe add collate function so that padding can be reduced/removed
    dataLoader = DataLoader(dataset, batch_size=64)

    return dataLoader
