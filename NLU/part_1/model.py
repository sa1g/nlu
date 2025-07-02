import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        bidirectional=False,
        emb_dropout=0.0,
        out_dropout=0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_dropout = nn.Dropout(out_dropout)
        # Dropout layer How/Where do we apply it?

        bidirectional_multiplier = 2 if bidirectional else 1
        # If bidirectional, we need to multiply the hidden size by 2
        hid_size *= bidirectional_multiplier

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        self.bidirectional = bidirectional

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        utt_emb = self.emb_dropout(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        # Process the batch
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get the last hidden state
        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1, :, :]

        utt_encoded = self.out_dropout(utt_encoded)
        last_hidden = self.out_dropout(last_hidden)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0, 2, 1)
        return slots, intent
