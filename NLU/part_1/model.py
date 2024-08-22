"""
In PyTorch the definition of a neural network is quite flexible. In ```__init__``` the layer that is going to be used are instantiated. In ```forward```, the architecture of the neural network is defined. Here you can find all the layers provided by Pytorch https://pytorch.org/docs/stable/nn.html while here you can find the recurrent layers https://pytorch.org/docs/stable/nn.html#recurrent-layers. 

<br><br>
**pack_padded_sequence** and **pad_packed_sequences** respectively compress and uncompress sequences to remove the padding embeddings from the computation, reducing the computational cost and, therefore, the CO2 emissions.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    def __init__(self, model_config, vocab_len, name, pad_index=0):
        super().__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.config = model_config
        self.name = name
        self.embedding = nn.Embedding(
            vocab_len, model_config["emb_size"], padding_idx=pad_index
        )

        self.utt_encoder = nn.LSTM(
            input_size=model_config["emb_size"],
            hidden_size=model_config["hid_size"],
            num_layers=model_config["n_layers"],
            bidirectional=model_config["bidirectional"],
            dropout=model_config["in_dropout"],
            batch_first=True,
        )

        bidirectional_multiplier = 2 if model_config["bidirectional"] else 1
        self.slot_out = nn.Linear(model_config["hid_size"] * bidirectional_multiplier, model_config["out_slot"])
        self.intent_out = nn.Linear(model_config["hid_size"] * bidirectional_multiplier, model_config["out_int"])

        self.dropout = nn.Dropout(self.config['emb_dropout'])

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)

        if self.config['emb_dropout'] > 0:
            utt_emb = self.dropout(utt_emb)

        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.config["bidirectional"]:
            # Concatenate the final states from both directions
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]

        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0, 2, 1)

        return slots, intent
