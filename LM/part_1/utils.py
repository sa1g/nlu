# Add functions or classes used for data loading and preprocessing

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Type, Union

import torch
import torch.utils.data as data
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader

from model import LM_LSTM, LM_RNN


@dataclass(frozen=True)
class Common:
    """
    Common configuration for the project
    """

    dataset_base_path: str = "../dataset/PennTreeBank/"
    train_batch_size: int = 256
    eval_batch_size: int = 256
    test_batch_size: int = 256


@dataclass
class ExperimentConfig:
    """
    Configuration for experiments
    """

    name: str = "Baseline"
    hid_size: int = 200
    emb_size: int = 300
    lr: float = 0.5
    clip: int = 5
    n_epochs: int = 100
    patience: int = 3

    dropout_embedding: float = 0
    dropout_output: float = 0

    model_type: Union[Type[LM_RNN], Type[LM_LSTM]] = LM_RNN
    optim: Union[Type[SGD], Type[AdamW]] = SGD


def read_file(path, eos_token="<eos>"):
    """
    Loading the corpus from the file
    Each line is a sentence, we add the end of sentence token to each line
    Args:
        path (str): Path to the file with the corpus
        eos_token (str): End of sentence token to be added to each line
    Returns:
        output (list): List of sentences with the end of sentence token added
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def get_vocab(corpus, special_tokens=[]):
    """
    Vocab with tokens to ids

    Args:
        corpus (list): List of sentences
        special_tokens (list): List of special tokens to be added to the vocab
    Returns:
        output (dict): Dictionary with tokens as keys and ids as values
        The ids are assigned in the order of appearance in the corpus
        and special tokens are assigned first
    """
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class Lang:
    def __init__(self, corpus, special_tokens=[]):
        """
        This class computes and stores our vocab. Word to ids and ids to word.
        Args:
            corpus (list): List of sentences
            special_tokens (list): List of special tokens to be added to the vocab

        Values:
            self.word2id (dict): Dictionary with tokens as keys and ids as values
            self.id2word (dict): Dictionary with ids as keys and tokens as values
            The ids are assigned in the order of appearance in the corpus
            and special tokens are assigned first
        """
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        """
        Vocab with tokens to ids
        Args:
            corpus (list): List of sentences
            special_tokens (list): List of special tokens to be added to the vocab
        Returns:
            output (dict): Dictionary with tokens as keys and ids as values
        """
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        """
        Args:
            corpus (list): List of sentences
            lang (Lang): Instance of Lang class with word2id and id2word mappings
        """
        super().__init__()

        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(
                sentence.split()[0:-1]
            )  # We get from the first token till the second-last token
            self.target.append(
                sentence.split()[1:]
            )  # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {"source": src, "target": trg}
        return sample

    def mapping_seq(self, data, lang):
        """
        Map sequences of tokens to corresponding ids using the word2id mapping
        Args:
            data (list): List of sentences, each sentence is a list of tokens
            lang (Lang): Instance of Lang class with word2id mapping
        Returns:
            res (list): List of sentences, each sentence is a list of ids corresponding to the tokens
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    logger.critical("OOV found!")
                    logger.critical(
                        "You have to deal with that"
                    )  # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res


def collate_fn(data, pad_token, device: torch.device):
    """
    Collate function for DataLoader
    It merges a list of samples to a batch of samples
    Args:
        data (list): List of samples, each sample is a dictionary with "source" and "target" keys
        pad_token (int): Padding token id, used to pad sequences to the same length
    Returns:
        new_item (dict): Dictionary with "source" and "target" keys, each key contains a tensor of shape (batch_size, max_len)
        "number_tokens" key contains the total number of tokens in the batch
    """

    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def get_dataloaders_and_lang(common: Common, device: torch.device):
    train_raw = read_file(os.path.join(common.dataset_base_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(common.dataset_base_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(common.dataset_base_path, "ptb.test.txt"))

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),
    )

    return train_loader, dev_loader, test_loader, lang


class EmojiFormatter(logging.Formatter):
    def format(self, record):
        # Add emoji based on log level
        if record.levelno == logging.INFO:
            emoji = "🤓\t"
        elif record.levelno == logging.WARNING:
            emoji = "⚠️\t"
        else:
            emoji = ""  # Default (no emoji for other levels)

        # Update the format string dynamically
        self._style._fmt = f"{emoji}%(levelname)s - %(asctime)s - %(message)s"
        return super().format(record)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Replace the default formatter with our custom one
formatter = EmojiFormatter(datefmt="%Y-%m-%d %H:%M:%S")  # Optional: Customize timestamp
for handler in logger.handlers:
    handler.setFormatter(formatter)
