# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functools import partial
from typing import Optional
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim

from tqdm import tqdm
import copy
# Import everything from functions.py file
# from functions import *

from utils import Common, ExperimentConfig, read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_RNN
from functions import experiments_launcher, train_loop, eval_loop, init_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_conf = ExperimentConfig()
    common = Common()

    experiments_launcher(experiment_config = [exp_conf], common=common,device=device)
