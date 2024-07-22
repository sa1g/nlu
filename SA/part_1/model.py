from transformers import BertModel
import torch


class SlotModel(torch.nn.Module):
    def __init__(self, slot_len):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.slot_classifier = torch.nn.Linear(self.bert.config.hidden_size, slot_len)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state

        slot_logits = self.slot_classifier(sequence_output)

        return slot_logits
