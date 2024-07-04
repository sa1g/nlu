import torch.nn as nn

class IntentSlotModel(nn.Module):
    def __init__(self, bert_model, slot_len, intent_len):
        super().__init__()
        self.bert = bert_model

        self.bert.resize_token_embeddings(30665)

        self.intent_classifier = nn.Linear(
            bert_model.config.hidden_size, intent_len
        )
        self.slot_classifier = nn.Linear(bert_model.config.hidden_size, slot_len)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits