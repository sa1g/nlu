# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
from utils import evaluate_ote
from model import SlotModel
from tqdm import tqdm


def train_loop(
    model: SlotModel,
    train_dataloader,
    optimizer,
    slot_loss_fn,
    slots2id,
    scheduler=None,
    grad_clip=False,
):
    model.train()

    loss_array = []
    batch_tqdm = tqdm(
        enumerate(train_dataloader), desc=f"Batch | Loss: {0:.4f}", leave=False
    )

    for _, data in batch_tqdm:
        optimizer.zero_grad()
        slot_logits = model(
            data["input_ids"], data["attention_mask"], data["token_type_ids"]
        )

        loss = slot_loss_fn(slot_logits.view(-1, len(slots2id)), data["slots"].view(-1))

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_tqdm.set_description(f"Batch | Loss: {loss.item():.4f}")
        loss_array.append(loss.item())

    return loss_array

# def eval_loop(
#     model : SlotModel, 
#     dataloader, 
#     slot_loss_fn, 
#     tokenizer, 
#     id2slots, 
#     slots2id
# ):
#     model.eval()
#     total_loss =  []

#     with torch.no_grad():
        
#         slot_hyp = []
#         for data in dataloader:
#             input_ids = data["input_ids"]
#             attention_mask = data["attention_mask"]
#             token_type_ids = data["token_type_ids"]
#             slots = data["slots"]
#             slots_len = data["slots_len"]

#             slot_logits = model(input_ids, attention_mask, token_type_ids)
#             loss = slot_loss_fn(slot_logits.view(-1, len(slots2id)), slots.view(-1)).item()
#             total_loss.append(loss)

#             slot_hyp.extend(torch.argmax(slot_logits, dim=2).cpu().tolist())

#         ot_precision, ot_recall, ot_f1 = evaluate_ote(data["slots"].to('cpu'), slot_hyp)
#     return ot_f1, ot_precision, ot_recall, total_loss

def eval_loop(
    model, 
    dataloader, 
    slot_loss_fn, 
    tokenizer, 
    id2slots, 
    slots2id
):
    model.eval()
    total_loss = []

    all_true_slots = []
    all_pred_slots = []

    with torch.no_grad():
        
        for data in dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            slots = data["slots"]
            slots_len = data["slots_len"]

            slot_logits = model(input_ids, attention_mask, token_type_ids)
            loss = slot_loss_fn(slot_logits.view(-1, len(slots2id)), slots.view(-1)).item()
            total_loss.append(loss)

            # Extract predictions
            slot_preds = torch.argmax(slot_logits, dim=2).cpu()
            true_slots = slots.cpu()

            for i in range(input_ids.size(0)):
                seq_len = slots_len[i]
                tmp_ref = [id2slots[true_slots[i][j].item()] for j in range(seq_len)]
                tmp_hyp = [id2slots[slot_preds[i][j].item()] for j in range(seq_len)]

                all_true_slots.append(tmp_ref)
                all_pred_slots.append(tmp_hyp)

    ot_precision, ot_recall, ot_f1 = evaluate_ote(all_true_slots, all_pred_slots)
    
    return ot_f1, ot_precision, ot_recall, total_loss