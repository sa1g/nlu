import numpy as np
import torch
from functions import NTAvSGD, experiments_launcher
from torch.optim import SGD
from utils import Common, ExperimentConfig

# Seeding
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = Common()

    experiments = [
        ExperimentConfig(
            name="wt20",
            lr=2.0,
            hid_size=300,
            weight_tying=True,
            optim=SGD,
        ),
        ExperimentConfig(
            name="wt30",
            lr=3.0,
            hid_size=300,
            weight_tying=True,
            optim=SGD,
        ),
        ExperimentConfig(
            name="wt40",
            lr=4.0,
            hid_size=300,
            weight_tying=True,
            optim=SGD,
        ),
        ExperimentConfig(
            name="30DropEmb",
            lr=3.0,
            hid_size=300,
            weight_tying=True,
            dropout_embedding=0.5,
            optim=SGD,
        ),
        ExperimentConfig(
            name="20DropOut",
            lr=2.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=SGD,
        ),
        ExperimentConfig(
            name="30DropOut",
            lr=3.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=SGD,
        ),
        ExperimentConfig(
            name="40DropOut",
            lr=4.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=SGD,
        ),
        ExperimentConfig(
            name="50DropOut",
            lr=5.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=SGD,
        ),
        ExperimentConfig(
            name="50NTAvSGD",
            lr=5.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=NTAvSGD,
            patience=5,
            non_monotonic_interval=3,
        ),
    ]

    experiments_launcher(experiment_config=experiments, common=common, device=device)
