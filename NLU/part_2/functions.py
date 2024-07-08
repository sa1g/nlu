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
            # input_ids, attention_mask, intent_labels, slot_labels = data
            # intent_labels = intent_labels.squeeze(1)

            # Forward
            intent_logits, slot_logits = model(data["input_ids"], data["attention_mask"], data["token_type_ids"])
            total_loss.append(
                calculate_loss(
                    intent_loss_fn=intent_loss_fn,
                    intent_logits=intent_logits,
                    intent_labels=data["intent"],
                    slot_loss_fn=slot_loss_fn,
                    slot_logits=slot_logits,
                    slot_labels=data["slots"],
                    slots2id=slots2id
                ).item()
            )

            intent_hyp = torch.argmax(intent_logits, dim=1)
            slot_hyp = torch.argmax(slot_logits, dim=2)

            # Intent inference
            ref_intents.extend(data["intent"].to("cpu").tolist())
            hyp_intents.extend(intent_hyp.to("cpu").tolist())
                    
            # Slot filling inference
            input_ids = data["input_ids"].to("cpu").tolist()
            if data["slots"].shape != slot_hyp.shape and data["slots"].shape != input_ids.shape:
                print("Shape mismatch")
                print(data["slots"].shape)
                print(slot_hyp.shape)
                print(input_ids.shape)
                exit()

            for input, s_ref, s_hyp, seq_length in zip(input_ids, data["slots"], slot_hyp, data["slots_len"]):
                tmp_ref = []
                tmp_hyp = []

                utterance = tokenizer.tokenize(tokenizer.decode(input, include_special_tokens=False))[:seq_length]

                for u, r,h in zip(utterance, s_ref, s_hyp):
                    tmp_ref.append((u, f"{id2slots[r.item()]}"))
                    tmp_hyp.append((u, f"{id2slots[h.item()]}"))

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