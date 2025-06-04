import torch
from functions import experiments_launcher
from model import LM_RNN, LM_LSTM
from torch.optim import SGD, AdamW
from utils import Common, ExperimentConfig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_conf = ExperimentConfig()
    common = Common()

    experiments_launcher(experiment_config=[exp_conf], common=common, device=device)
