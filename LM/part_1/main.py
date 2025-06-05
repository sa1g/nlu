import torch
from functions import experiments_launcher
from model import LM_LSTM, LM_RNN
from torch.optim import SGD, AdamW
from utils import Common, ExperimentConfig

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = Common()

    experiments = [
        ExperimentConfig(name="Baseline05", model_type=LM_RNN, lr=0.5, n_epochs=2),
        ExperimentConfig(
            name="Baseline10",
            model_type=LM_RNN,
            lr=1.0,
        ),
        ExperimentConfig(
            name="Baseline20",
            model_type=LM_RNN,
            lr=2.0,
        ),
        ExperimentConfig(
            name="LSTM05",
            model_type=LM_LSTM,
            lr=0.5,
        ),
        ExperimentConfig(
            name="LSTM10",
            model_type=LM_LSTM,
            lr=1.0,
        ),
        ExperimentConfig(
            name="LSTM20",
            model_type=LM_LSTM,
            lr=2.0,
        ),
        # ExperimentConfig(
        #     name="LSTM10-DropEmb-05",
        #     model_type=LM_LSTM,
        #     lr=1.0,
        #     dropout_embedding=0.5,
        # ),
        # ExperimentConfig(
        #     name="LSTM10-DropOut-05",
        #     model_type=LM_LSTM,
        #     lr=1.0,
        #     dropout_output=0.5,
        # ),
        # ExperimentConfig(
        #     name="LSTM10-Drop-05",
        #     model_type=LM_LSTM,
        #     lr=1.0,
        #     dropout_output=0.5,
        #     dropout_embedding=0.5,
        # ),
        # ExperimentConfig(
        #     name="LSTM10-Drop-05-AdamW",
        #     model_type=LM_LSTM,
        #     lr=1.0,
        #     dropout_output=0.5,
        #     dropout_embedding=0.5,
        #     optim=AdamW,
        # ),
    ]

    # exp_conf = ExperimentConfig(
    #     model_type=LM_LSTM,
    #     dropout_embedding=0.5,
    #     dropout_output=0.5,
    #     optim=AdamW,
    #     lr=0.001,
    # )

    experiments_launcher(experiment_config=experiments, common=common, device=device)
