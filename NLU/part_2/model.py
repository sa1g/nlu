from transformers import BertModel
import torch


class IntentSlotModel(torch.nn.Module):
    def __init__(self, slot_len, intent_len):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.intent_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, intent_len
        )
        self.slot_classifier = torch.nn.Linear(self.bert.config.hidden_size, slot_len)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits
