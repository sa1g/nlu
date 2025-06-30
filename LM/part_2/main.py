import numpy as np
import torch
from functions import NTAvSGD, experiments_launcher
from utils import Common, ExperimentConfig

# Seeding
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = Common()

    experiments = [
        # ExperimentConfig(
        #     name="variational_dropout_test",
        #     lr=0.5,
        #     n_epochs=2,
        #     emb_size=300,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_embedding=0.2,
        #     dropout_output=0.2,
        #     optim=NTAvSGD,
        # ),
        # ExperimentConfig(
        #     name="wt20",
        #     lr=2.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="wt30",
        #     lr=3.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="wt40",
        #     lr=4.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="30DropEmb",
        #     lr=3.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_embedding=0.5,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="20DropOut",
        #     lr=2.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="30DropOut",
        #     lr=3.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="40DropOut",
        #     lr=4.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=SGD,
        # ),
        # ExperimentConfig(
        #     name="50DropOut",
        #     lr=5.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=SGD,
        # ),
        ## TODO: add ntavsgd experiments (play with lr)
        ExperimentConfig(
            name="provaaaaaaaNTA",
            lr=5.0,
            hid_size=300,
            weight_tying=True,
            dropout_output=0.5,
            optim=NTAvSGD,
            n_epochs=150,
        ),
        # ExperimentConfig(
        #     name="40NTAvSGD",
        #     lr=4.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=NTAvSGD,
        #     n_epochs=150,
        # ),
        # ExperimentConfig(
        #     name="50NTAvSGD",
        #     lr=5.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=NTAvSGD,
        #     n_epochs=150,
        # ),
        # ExperimentConfig(
        #     name="400SGD",
        #     lr=40.0,
        #     hid_size=300,
        #     weight_tying=True,
        #     dropout_output=0.5,
        #     optim=SGD,
        #     n_epochs=150,
        # ),
    ]

    experiments_launcher(experiment_config=experiments, common=common, device=device)
