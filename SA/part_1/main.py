import os

import torch
from functions import experiment_launcher
from utils import Common, ExperimentConfig

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
    PAD_TOKEN = 0

    common = Common()

    experiment_config = [
        ExperimentConfig(name="5e-5", lr=5e-5, n_epochs=10),
        ExperimentConfig(name="3e-5", lr=3e-5, n_epochs=10),
        ExperimentConfig(name="2e-5", lr=2e-5, n_epochs=10),
        ExperimentConfig(name="5e-5-Sch", lr=5e-5, scheduler=True, n_epochs=10),
        ExperimentConfig(name="5e-5-Clip", lr=5e-5, grad_clip=True, n_epochs=10),
        ExperimentConfig(
            name="5e-5-SchClip", lr=5e-5, scheduler=True, grad_clip=True, n_epochs=10
        ),
    ]

    experiment_launcher(experiment_config, common, device)
