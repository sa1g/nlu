import torch
import torch.nn as nn
import logging


class LM_RNN(nn.Module):
    def __init__(self, config: dict):
        """
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: int = 0,
        out_dropout: int = 0,
        n_layers: int = 1,
        """

        super().__init__()

        logging.debug("LM_RNN")

        self.embedding = nn.Embedding(
            config["output_size"], config["emb_size"], padding_idx=config["pad_index"]
        )
        self.emb_dropout = nn.Dropout(config["emb_dropout"])
        self.rnn = nn.RNN(config["emb_size"], config["hid_size"], config["n_layers"])
        self.out_dropout = nn.Dropout(config["out_dropout"])
        self.output = nn.Linear(config["hid_size"], config["output_size"])

        self.name = f"{self.__class__.__name__}_emb_{config['emb_size']}_hid_{config['hid_size']}_edr_{config['emb_dropout']}_odr_{config['out_dropout']}_lay_{config['n_layers']}"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn, _ = self.rnn(emb)
        out = self.out_dropout(rnn)
        out = self.output(out)
        out = out.permute(0, 2, 1)
        return out


class LM_LSTM(nn.Module):
    def __init__(self, config: dict):
        """
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: int = 0,
        out_dropout: int = 0,
        n_layers: int = 1,
        """
        super().__init__()

        logging.debug("LM_LSTM")

        self.embedding = nn.Embedding(
            config["output_size"], config["emb_size"], padding_idx=config["pad_index"]
        )
        self.emb_dropout = nn.Dropout(config["emb_dropout"])
        self.rnn = nn.LSTM(
            config["emb_size"],
            config["hid_size"],
            config["n_layers"],
            bidirectional=False,
        )
        self.out_dropout = nn.Dropout(config["out_dropout"])
        self.output = nn.Linear(config["hid_size"], config["output_size"])

        self.name = f"{self.__class__.__name__}_emb_{config['emb_size']}_hid_{config['hid_size']}_edr_{config['emb_dropout']}_odr_{config['out_dropout']}_lay_{config['n_layers']}"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn, _ = self.rnn(emb)
        out = self.out_dropout(rnn)
        out = self.output(out)
        out = out.permute(0, 2, 1)
        return out


class LM_LSTM_WS(nn.Module):
    def __init__(self, config: dict):
        """
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: int = 0,
        out_dropout: int = 0,
        n_layers: int = 1,
        """
        super().__init__()

        logging.debug("LM_LSTM_WS")

        self.embedding = nn.Embedding(
            config["output_size"], config["emb_size"], padding_idx=config["pad_index"]
        )
        self.emb_dropout = nn.Dropout(config["emb_dropout"])
        self.rnn = nn.LSTM(
            config["emb_size"],
            config["hid_size"],
            config["n_layers"],
            bidirectional=False,
        )
        self.out_dropout = nn.Dropout(config["out_dropout"])
        self.output = nn.Linear(config["hid_size"], config["output_size"])
        self.softmax = nn.Softmax(-1)

        self.softmax.weight = self.embedding.weight

        self.name = f"{self.__class__.__name__}_emb_{config['emb_size']}_hid_{config['hid_size']}_edr_{config['emb_dropout']}_odr_{config['out_dropout']}_lay_{config['n_layers']}"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn, _ = self.rnn(emb)
        out = self.out_dropout(rnn)
        out = self.output(out)
        out = self.softmax(out).permute(0, 2, 1)

        return out


class LM_LSTM_VD(nn.Module):
    """
    W/variational dropout
    """

    def __init__(self, config):
        """
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        variational_dropout
        n_layers: int = 1,
        """
        super().__init__()

        logging.debug("LM_LSTM_VD")

        self.embedding = nn.Embedding(
            config["output_size"], config["emb_size"], padding_idx=config["pad_index"]
        )
        self.rnn = nn.LSTM(
            config["emb_size"],
            config["hid_size"],
            config["n_layers"],
            bidirectional=False,
        )
        self.output = nn.Linear(config["hid_size"], config["output_size"])

        self.name = f"{self.__class__.__name__}_emb_{config['emb_size']}_hid_{config['hid_size']}_vdr_{config['variational_dropout']}_lay_{config['n_layers']}"

        self.variational_dropout = config["variational_dropout"]
        self.mask = 0

    def variational_dropout(self, x):
        mask = self.mask
        mask = mask.expand_as(x)
        return x * mask

    def forward(self, input_sequence):
        self.mask = input_sequence.new(
            (input_sequence.size(0), 1, input_sequence(2))
            .bernulli_(1 - self.variational_dropout)
            .div_(1 - self.variational_dropout)
        )

        emb = self.embedding(input_sequence)

        emb_dropout = self.variational_dropout(emb)
        rnn, _ = self.rnn(emb_dropout)

        rnn_dropout = self.variational_dropout(rnn)
        out = self.output(rnn_dropout).permute(0, 2, 1)

        return out
