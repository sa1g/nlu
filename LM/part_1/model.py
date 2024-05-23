import torch
import torch.nn as nn
import logging

def variational_dropout(x, mask):
    mask = mask.expand_as(x)
    return x * mask

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
        assert not ((config["emb_dropout"] or config["out_dropout"]) and config["variational_dropout"])

        self.config = config

        if (config["emb_dropout"] != 0):
            logging.debug("emb dropout activated")
        if (config["out_dropout"] != 0):
            logging.debug("out dropout activated")


        super().__init__()

        self.name = f"{self.__class__.__name__}_emb_{config['emb_size']}_hid_{config['hid_size']}_edr_{config['emb_dropout']}_odr_{config['out_dropout']}_lay_{config['n_layers']}_weight_tying_{config['weight_tying']}__variational_dropout_{config['variational_dropout']}_optim_{config['optim_name']}"
        logging.debug("LM_LSTM")

        self.embedding = nn.Embedding(
            config["output_size"], config["emb_size"], padding_idx=config["pad_index"]
        )
        # Dropout can be either activated or not
        if self.config['emb_dropout'] != 0:
            self.emb_dropout = nn.Dropout(config["emb_dropout"])
        
        # LSTM model
        self.rnn = nn.LSTM(
            config["emb_size"],
            config["hid_size"],
            config["n_layers"],
            bidirectional=False,
        )

        # Dropout can be either activated or not
        if self.config['out_dropout'] != 0:
            self.out_dropout = nn.Dropout(config["out_dropout"])
        
        self.output = nn.Linear(config["hid_size"], config["output_size"])
        
        # if there's variational dropout manage it
        if config["variational_dropout"] != 0:
            self.var_dropout_perc = config["variational_dropout"]


        # Enable/disable weight share
        if config["weight_tying"]:
            assert config['emb_size'] == config['hid_size']
            self.output.weight = self.embedding.weight


    def forward(self, input_sequence):
        if self.config['variational_dropout'] != 0:
            mask = input_sequence.new(
            (input_sequence.size(0), 1, input_sequence(2))
            .bernulli_(1 - self.var_dropout_perc)
            .div_(1 - self.var_dropout_perc)
        )

        emb = self.embedding(input_sequence)
        
        if self.config['emb_dropout'] != 0:
            emb = self.emb_dropout(emb)

        if self.config['variational_dropout'] != 0:
            emb = variational_dropout(emb, mask)

        rnn, _ = self.rnn(emb)
        
        if self.config['out_dropout'] != 0:
            rnn = self.out_dropout(rnn)

        if self.config['variational_dropout'] != 0:
            rnn = variational_dropout(rnn, mask)

        out = self.output(rnn)
        out = out.permute(0, 2, 1)
        return out
