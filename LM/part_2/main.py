import torch
from torch.optim import SGD

from functions import experiments_launcher, NTAvSGD
from model import LM_LSTM
from utils import Common, ExperimentConfig

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = Common()

    experiments = [
        ExperimentConfig(
            name="variational_dropout_test",
            lr=0.5,
            n_epochs=2,
            emb_size=300,
            hid_size=300,
            weight_tying=True,
            dropout_embedding=0.2,
            dropout_output=0.2,
            optim=NTAvSGD,
        ),
    ]

    experiments_launcher(experiment_config=experiments, common=common, device=device)
