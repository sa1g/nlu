# Global variables
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
        ExperimentConfig(
            name="dwoaindowaisndoisanoi",
            lr=0.0001,
        ),
        # ExperimentConfig(
        #     name="Baseline0001",
        #     lr=0.0001,
        # ),
        # ExperimentConfig(
        #     name="Baseline001",
        #     lr=0.001,
        # ),
        # ExperimentConfig(
        #     name="Baseline01",
        #     lr=0.01,
        # ),
        # ExperimentConfig(
        #     name="DropEmb001",
        #     emb_dropout=0.5,
        #     lr=0.001,
        # ),
        # ExperimentConfig(
        #     name="DropOut001",
        #     out_dropout=0.5,
        #     lr=0.001,
        # ),
        # ExperimentConfig(
        #     name="Bidirectional001",
        #     bidirectional=True,
        #     lr=0.001,
        # ),
        # ExperimentConfig(
        #     name="BidirectionalDrop001",
        #     bidirectional=True,
        #     out_dropout=0.5,
        #     lr=0.001,
        # ),
    ]

    experiment_launcher(experiment_config, common, device)
