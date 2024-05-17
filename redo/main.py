from functions import (
    get_model,
    get_optimizer,
    init_weights,
    train,
    get_loaders_lang,
)
from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(format="%(msecs)s:%(levelname)s:%(message)s", level=logging.DEBUG)


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-p', default='dataset', help='Path to the dataset')
parser.add_argument('-trainbs', type=int, default=128, help='Training batch size')
parser.add_argument('-devbs', type=int, default=128, help='Development batch size')
parser.add_argument('-testbs', type=int, default=128, help='Test batch size')
parser.add_argument('-emb_size', type=int, default=300, help='Embedding size')
parser.add_argument('-hid_size', type=int, default=300, help='Hidden layer size')
parser.add_argument('-emb_dropout', type=float, default=0, help='Embedding dropout rate')
parser.add_argument('-out_dropout', type=float, default=0, help='Output layer dropout rate')
parser.add_argument('-n_layers', type=int, default=1, help='Number of layers')
parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-n_epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('-clip', type=float, default=5, help='Gradient clipping value')

if __name__ == "__main__":
    DEVICE = "cuda:0"

    args = parser.parse_args()
    dataset_path = args.p
    train_batch_size = args.trainbs
    dev_batch_size = args.devbs
    test_batch_size = args.testbs
    emb_size = args.emb_size
    hid_size = args.hid_size
    emb_dropout = args.emb_dropout
    out_dropout = args.out_dropout
    n_layers = args.n_layers
    lr = args.lr
    n_epochs = args.n_epochs
    clip = args.clip

    # dataset/ptb.test.txt
    train_loader, dev_loader, test_loader, lang = get_loaders_lang(
        dataset_path, train_batch_size, dev_batch_size, test_batch_size
    )

    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]
    output_size = vocab_len

    # MODEL SETUP
    model = get_model(
        emb_size=emb_size,
        hid_size=hid_size,
        output_size=output_size,
        pad_index=pad_index,
        emb_dropout=emb_dropout,
        out_dropout=out_dropout,
        n_layers=n_layers,
        device=DEVICE,
        init_weights=init_weights,
        model_type="LM_RNN",
    )

    print(type(model))
    

    optimizer = get_optimizer(model, optim_name="SGD", lr=lr)

    logging.debug("Model done")

    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{model.name}")

    config = {
        "emb_size": emb_size,
        "hid_size": hid_size,
        "emb_dropout": emb_dropout,
        "out_dropout": out_dropout,
        "n_layers": n_layers,
        "lr": lr,
        "train_batch_size": train_batch_size,
        "dev_batch_size": dev_batch_size,
        "test_batch_size": test_batch_size,
    }

    # TRAINING
    train(
        model=model,
        optimizer=optimizer,
        lang=lang,
        writer=writer,
        n_epochs=n_epochs,
        clip=clip,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=DEVICE,
    )
