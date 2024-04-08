import copy
import math
import numpy as np
import torch
from tqdm import tqdm
from utils import get_data_loaders
from model import LM_RNN
from functions import train_loop, eval_loop, init_weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# Rest of the code

# TODO: plot the loss and the perplexity
# TODO: tensorboard


if __name__ == "__main__":
    ############
    # Hyperparameters
    ############

    HIDDEN_SIZE = 200
    EMBEDDED_SIZE = 300
    LR = 1
    CLIP = 5
    N_EPOCHS = 3
    PATIENCE = 3
    train_batch_size = 32
    eval_batch_size = 32
    test_batch_size = 32

    # Tensorboard
    # can add a comment to the writer so that you can see some information in tensorboard
    writer = SummaryWriter(log_dir="runs/LM_RNN")
    DEVICE = "cuda:0"

    lang, train_loader, eval_loader, test_loader = get_data_loaders(
        train_batch_size, eval_batch_size, test_batch_size
    )
    VOCAB_LEN = len(lang.word2id)

    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step
    # With SGD try with an higher learning rate (> 1 for instance)

    ############
    # Model
    ############

    model = LM_RNN(
        EMBEDDED_SIZE, HIDDEN_SIZE, VOCAB_LEN, pad_index=lang.word2id["<pad>"]
    ).to(DEVICE)
    model.apply(init_weights)

    ############
    # Training
    ############

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=LR)
    # Define training loss
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    # Define evaluation loss
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    losses_train = []
    losses_eval = []
    sampled_epochs = []

    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(1, N_EPOCHS))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_eval, loss_eval = eval_loop(eval_loader, criterion_eval, model)
            losses_eval.append(np.asarray(loss_eval).mean())

            pbar.set_description(f"PPL: {ppl_eval}")

            writer.add_scalar("Loss/train", losses_train[-1], epoch)
            writer.add_scalar("Loss/eval", losses_eval[-1], epoch)
            writer.add_scalar("PPL/eval", ppl_eval, epoch)

            if ppl_eval < best_ppl:  # the lower, the better
                best_ppl = ppl_eval
                best_model = copy.deepcopy(model).to("cpu")
                PATIENCE = 3
            else:
                PATIENCE -= 1

            if PATIENCE <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test ppl: ", final_ppl)

    ############
    # Save the model
    ############

    # To save the model
    # path = 'model_bin/model_name.pt'
    # torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
    
    writer.close()
