import copy
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from functions import eval_loop, init_weights, train_loop
from model import LM_RNN
from functools import partial


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    
    DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu
    
    ############
    # Load Data
    ############


    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Vocab is computed only on training set 
    # We add two special tokens end of sentence and padding 
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    print(len(vocab))

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    ############
    # Hyperparameters
    ############

    # HIDDEN_SIZE = 300
    # EMBEDDED_SIZE = 300
    # N_LAYERS = 1
    # LR = 1.5
    # CLIP = 5
    # N_EPOCHS = 50
    # PATIENCE = 3
    # train_batch_size = 128
    # eval_batch_size = 512
    # test_batch_size = 128

    # EXPERIMENT_NAME = "lstm-do-0.1-0.1-lr-1.5-hs-300-batch-128"
    
    ############
    # Tensorboard
    ############
    
    # Tensorboard
    # can add a comment to the writer so that you can see some information in tensorboard
    # writer = SummaryWriter(log_dir=f"runs/{EXPERIMENT_NAME}")

    
    
    # # Save experiment config with tensorboard
    # config = {
    #     "HIDDEN_SIZE": HIDDEN_SIZE,
    #     "EMBEDDED_SIZE": EMBEDDED_SIZE,
    #     "N_LAYERS": N_LAYERS,
    #     "LR": LR,
    #     "CLIP": CLIP,
    #     "N_EPOCHS": N_EPOCHS,
    #     "PATIENCE": PATIENCE,
    #     "train_batch_size": train_batch_size,
    #     "eval_batch_size": eval_batch_size,
    #     "test_batch_size": test_batch_size,
    #     # "seed": "nan",
    # }

    # writer.add_text("Experiment Config", str(config))

    ############
    # Data
    ############
    
    # lang, train_loader, eval_loader, test_loader = get_data_loaders(
    #     train_batch_size, eval_batch_size, test_batch_size
    # )
    # VOCAB_LEN = len(lang.word2id)

    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step
    # With SGD try with an higher learning rate (> 1 for instance)

    ############
    # Model
    ############

    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step

    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 0.0001 # This is definitely not good for SGD
    clip = 5 # Clip the gradient
    device = 'cuda:0'

    vocab_len = len(lang.word2id)

    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    # model = LM_LSTM_DROPOUT(
    #     EMBEDDED_SIZE, HIDDEN_SIZE, VOCAB_LEN, pad_index=lang.word2id["<pad>"], n_layers=N_LAYERS, out_dropout=0.1,
    #     emb_dropout=0.1
    # ).to(DEVICE)
    # model.apply(init_weights)

    ############
    # Early stopping
    ############

    # early_stopper = EarlyStopper(PATIENCE, min_delta=10)

    # ############
    # # Training
    # ############

    # # Define optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LR)
    # # Define training loss
    # criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    # # Define evaluation loss
    # criterion_eval = nn.CrossEntropyLoss(
    #     ignore_index=lang.word2id["<pad>"], reduction="sum"
    # )

    # losses_train = []
    # losses_eval = []
    # sampled_epochs = []

    # best_ppl = math.inf
    # best_model = None

    # pbar = tqdm(range(1, N_EPOCHS))
    # for epoch in pbar:
    #     loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)

    #     if epoch % 1 == 0:
    #         sampled_epochs.append(epoch)
    #         losses_train.append(np.asarray(loss).mean())

    #         ppl_eval, loss_eval = eval_loop(eval_loader, criterion_eval, model)
    #         losses_eval.append(np.asarray(loss_eval).mean())

    #         pbar.set_description(f"PPL: {ppl_eval}")

    #         writer.add_scalar("Loss/train", losses_train[-1], epoch)
    #         writer.add_scalar("Loss/eval", losses_eval[-1], epoch)
    #         writer.add_scalar("PPL/eval", ppl_eval, epoch)

    #         print(
    #             f"Epoch: {epoch}, Eval PPL: {ppl_eval}, Train Loss: {losses_train[-1]}, Eval Loss: {losses_eval[-1]}")

    #         # if early_stopper.early_stop(ppl_eval):             
    #         #     break
    #         if ppl_eval < best_ppl:  # the lower, the better
    #             best_ppl = ppl_eval
    #             best_model = copy.deepcopy(model).to("cpu")
    #             PATIENCE = 3
    #         else:
    #             PATIENCE -= 1

    #         if PATIENCE <= 0:  # Early stopping with patience
    #             break  # Not nice but it keeps the code clean

    # final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    # print("Test ppl: ", final_ppl)

    # ############
    # # Save the model
    # ############

    # path = f'bin/{EXPERIMENT_NAME}.pt'
    # torch.save(model.state_dict(), path)
    
    # # To load the model you need to initialize it
    # # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # # Then you load it
    # # model.load_state_dict(torch.load(path))
    
    # writer.close()
