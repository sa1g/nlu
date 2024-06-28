# Add functions or classes used for data loading and preprocessing
import json
import os
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging

# Global variables
device = "cuda:0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
PAD_TOKEN = 0


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
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {"pad": PAD_TOKEN}
        if unk:
            vocab["unk"] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk="unk"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {"utterance": utt, "slots": slots, "intent": intent}
        return sample

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


def collate_fn(data):
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
        # We remove these tensors from the computational graph
        padded_seqs = padded_seqs.detach()
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


def split_dev_set(tmp_train_raw, test_raw):
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
    X_train, X_dev, y_train, y_dev = train_test_split(
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

    y_test = [x["intent"] for x in test_raw]

    # Dataset size
    logging.debug("Train size %i", len(train_raw))
    logging.debug("DEV size %i", len(dev_raw))
    logging.debug("TEST size %i", len(test_raw))

    return train_raw, dev_raw, test_raw


def get_loaders_lang(dataset_path, train_batch_size, dev_batch_size, test_batch_size):
    tmp_train_raw = load_data(os.path.join(dataset_path, "train.json"))
    test_raw = load_data(os.path.join(dataset_path, "test.json"))

    logging.debug("Train samples: %s", len(tmp_train_raw))
    logging.debug("Test samples: %s", len(test_raw))

    train_raw, dev_raw, test_raw = split_dev_set(tmp_train_raw, test_raw)

    # w2id = {"pad": PAD_TOKEN}  # Pad tokens is 0 so the index count should start from 1
    # slot2id = {
    #     "pad": PAD_TOKEN
    # }  # Pad tokens is 0 so the index count should start from 1
    # intent2id = {}

    # logging.debug(
    #     "# Vocabulary size: %i", len(w2id) - 2
    # )  # we remove pad and unk from the count
    # logging.debug("# Slots: %i", len(slot2id) - 1)
    # logging.debug("# Intent: %i", len(intent2id))


    w2id = {'pad':PAD_TOKEN, 'unk': 1}
    slot2id = {'pad':PAD_TOKEN}
    intent2id = {}
    # Map the words only from the train set
    # Map slot and intent labels of train, dev and test set. 'unk' is not needed.
    for example in train_raw:
        for w in example['utterance'].split():
            if w not in w2id:
                w2id[w] = len(w2id)   
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)
            
    for example in dev_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)
            
    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    logging.debug(
        "# Vocabulary size: %i", len(w2id) - 2
    )  # we remove pad and unk from the count
    logging.debug("# Slots: %i", len(slot2id) - 1)
    logging.debug("# Intent: %i", len(intent2id))

    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
    # however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=dev_batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, lang, w2id, slot2id, intent2id
