import numpy as np
import torch
from functions import experiments_launcher
from model import LM_LSTM, LM_RNN
from torch.optim import AdamW
from utils import Common, ExperimentConfig

# Seeding
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = Common()

    experiments = [
        ExperimentConfig(name="Baseline05", model_type=LM_RNN, lr=0.5),
        ExperimentConfig(name="Baseline05", model_type=LM_RNN, lr=0.5, hid_size=300),
        ExperimentConfig(name="Baseline10", model_type=LM_RNN, lr=1.0, hid_size=300),
        ExperimentConfig(name="Baseline20", model_type=LM_RNN, lr=2.0, hid_size=300),
        ExperimentConfig(name="LSTM05", model_type=LM_LSTM, lr=0.5, hid_size=300),
        ExperimentConfig(name="LSTM10", model_type=LM_LSTM, lr=1.0, hid_size=300),
        ExperimentConfig(name="LSTM20", model_type=LM_LSTM, lr=2.0, hid_size=300),
        ExperimentConfig(name="LSTM30", model_type=LM_LSTM, lr=3.0, hid_size=300),
        ExperimentConfig(name="LSTM40", model_type=LM_LSTM, lr=4.0, hid_size=300),
        ExperimentConfig(
            name="LSTM30-DropEmb-05",
            model_type=LM_LSTM,
            lr=3.0,
            dropout_embedding=0.5,
            hid_size=300,
        ),
        ExperimentConfig(
            name="LSTM30-DropOut-05",
            model_type=LM_LSTM,
            lr=3.0,
            dropout_output=0.5,
            hid_size=300,
        ),
        ExperimentConfig(
            name="LSTM30-Drop-05",
            model_type=LM_LSTM,
            lr=3.0,
            dropout_output=0.5,
            dropout_embedding=0.5,
            hid_size=300,
        ),
        ExperimentConfig(
            name="LSTM001-Drop-05-AdamW",
            model_type=LM_LSTM,
            lr=0.001,
            dropout_output=0.5,
            optim=AdamW,
            hid_size=300,
        ),
        ExperimentConfig(
            name="LSTM003-Drop-05-AdamW",
            model_type=LM_LSTM,
            lr=0.003,
            dropout_output=0.5,
            optim=AdamW,
            hid_size=300,
        ),
    ]

    experiments_launcher(experiment_config=experiments, common=common, device=device)
