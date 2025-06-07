# Global variables
import os

import torch

from functions import experiment_launcher
from utils import Common, ExperimentConfig

# words = sum(
#     [x["utterance"].split() for x in train_raw], []
# )  # No set() since we want to compute
# # the cutoff
# corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels,
# # however this depends on the research purpose
# slots = set(sum([line["slots"].split() for line in corpus], []))
# intents = set([line["intent"] for line in corpus])

# lang = Lang(words, intents, slots, cutoff=0)


# import torch
# import torch.utils.data as data


# from torch.utils.data import DataLoader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
    PAD_TOKEN = 0

    common = Common()

    experiment_config = [
        ExperimentConfig(out_dropout=0.5, emb_dropout=0.5, bidirectional=True)
    ]

    # ###########################
    experiment_launcher(experiment_config, common, device)
