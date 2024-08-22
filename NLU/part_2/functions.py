from sklearn.metrics import classification_report
from tqdm import tqdm
from conll import evaluate
import torch
from model import IntentSlotModel

def calculate_loss(
    intent_loss_fn, slot_loss_fn, intent_logits, slot_logits, intent_labels, slot_labels, slots2id
):
    intent_loss = intent_loss_fn(intent_logits, intent_labels)
    slot_loss = slot_loss_fn(slot_logits.view(-1, len(slots2id)), slot_labels.view(-1))

    return intent_loss + slot_loss

def eval_loop(
    model: IntentSlotModel,
    dataloader,
    intent_loss_fn,
    slot_loss_fn,
    tokenizer, id2slots, slots2id
):
    model.eval()
    total_loss = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            intent_labels = data["intent"]
            slot_labels = data["slots"]
            slots_len = data["slots_len"]

            # Forward pass
            intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids)
            loss = calculate_loss(
                intent_loss_fn=intent_loss_fn,
                intent_logits=intent_logits,
                intent_labels=intent_labels,
                slot_loss_fn=slot_loss_fn,
                slot_logits=slot_logits,
                slot_labels=slot_labels,
                slots2id=slots2id
            ).item()
            total_loss.append(loss)

            # Intent inference
            intent_hyp = torch.argmax(intent_logits, dim=1)
            ref_intents.extend(intent_labels.cpu().tolist())
            hyp_intents.extend(intent_hyp.cpu().tolist())

            # Slot filling inference
            slot_hyp = torch.argmax(slot_logits, dim=2)

            for i in range(input_ids.size(0)):
                seq_length = slots_len[i]
                utterance = tokenizer.tokenize(tokenizer.decode(input_ids[i].cpu().tolist(), include_special_tokens=False))[:seq_length]

                tmp_ref = [(utterance[j], id2slots[slot_labels[i][j].item()]) for j in range(seq_length)]
                tmp_hyp = [(utterance[j], id2slots[slot_hyp[i][j].item()]) for j in range(seq_length)]

                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)

        f1_slot = evaluate(ref_slots, hyp_slots)

        accuracy_intention = classification_report(
            ref_intents,
            hyp_intents,
            output_dict=True,
            zero_division=False,
        )['accuracy']

        return accuracy_intention, f1_slot["total"]["f"], total_loss

def train_loop(
    model: IntentSlotModel,
    train_dataloader,
    optimizer,
    intent_loss_fn,
    slot_loss_fn,
    slots2id,
    scheduler = None,
    grad_clip = False
):
    model.train()

    loss_array = []
    batch_tqdm = tqdm(enumerate(train_dataloader), desc=f"Batch | Loss: {0:.4f}", leave=False)

    for _, data in batch_tqdm:
        optimizer.zero_grad()
        intent_logits, slot_logits = model(data["input_ids"], data["attention_mask"], data["token_type_ids"])

        loss = calculate_loss(
            intent_loss_fn=intent_loss_fn,
            intent_logits=intent_logits,
            intent_labels=data["intent"],
            slot_loss_fn=slot_loss_fn,
            slot_logits=slot_logits,
            slot_labels=data["slots"],
            slots2id=slots2id
        )

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        batch_tqdm.set_description(f"Batch | Loss: {loss.item():.4f}")
        loss_array.append(loss.item())

    return loss_array