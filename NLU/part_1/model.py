"""
In PyTorch the definition of a neural network is quite flexible. In ```__init__``` the layer that is going to be used are instantiated. In ```forward```, the architecture of the neural network is defined. Here you can find all the layers provided by Pytorch https://pytorch.org/docs/stable/nn.html while here you can find the recurrent layers https://pytorch.org/docs/stable/nn.html#recurrent-layers. 

<br><br>
**pack_padded_sequence** and **pad_packed_sequences** respectively compress and uncompress sequences to remove the padding embeddings from the computation, reducing the computational cost and, therefore, the CO2 emissions.
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, model_config, vocab_len, name, pad_index=0):
        super().__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.name = name
        self.embedding = nn.Embedding(
            vocab_len, model_config["emb_size"], padding_idx=pad_index
        )
        
        # dropout: float = 0,
        self.utt_encoder = nn.LSTM(
            input_size=model_config["emb_size"],
            hidden_size=model_config["hid_size"],
            num_layers=model_config["n_layers"],
            bidirectional=model_config["bidirectional"],
            batch_first=True,
        )
        self.slot_out = nn.Linear(model_config["hid_size"], model_config["out_slot"])
        self.intent_out = nn.Linear(model_config["hid_size"], model_config["out_int"])

        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = self.embedding(utterance)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Get the last hidden state
        last_hidden = last_hidden[-1, :, :]

        # Is this another possible way to get the last hidden state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)

        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss

        # Slot size: batch_size, classes, seq_len
        return slots, intent
