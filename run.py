import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from  evaluation.core import State
from typing import Dict, List
import datetime
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
from copy import deepcopy
import random
from Rushi.PPO.trainer import PPOTrainer
from Rushi.PPO.ml import PPOAgent
from Rushi.PPO.core import EarlyEnvironment, EarlyEnvs
from Rushi.PPO.utils import Logger, makeFeatures, getInvalidActionMask, getStateFeatures, getStockFeatures, interpolatorNotForDay,addPatterns
from evaluation.core import BaseStrategy
from evaluation.constants import HOLD_ACTION,BUY_ACTION,SELL_ACTION
from Rushi.PPO.constants import MAX_HOLDING, MAX_TRADE_TIME_IN_5_MINUTE, MAX_TRADE_TIME_IN_DAYS,ROLLING_WINDOW_SIZE_IN_DAYS, MAX_TRADE_TIME_IN_MINUTE, MODES, minutes_5_in_day, minutes_in_day, ROLLING_WINDOW_SIZE_IN_MINUTE, ROLLING_WINDOW_SIZE_IN_5_MINUTE,end_time_5_minute, end_time_minute
from Rushi.PPO.constants import end_time_minute, end_time_5_minute
from Rushi.PPO.ppoStrategy import PPOStrategy
from evaluation.backTest import BackTest
from tqdm import tqdm
from Rushi.PPO.rollingTrainer import RollingTrain
from QLearning.rollingTrain import RollingTrain as QRollingTrain



ppo_config = dict(
    num_envs = 128,
    num_steps = 256,
    observation_dim = 13 + 4,
    action_space_dim = 3,
    gamma = .9,
    num_mini_batches = 4,
    update_epochs = 4,
    clip_coef = .2,
    entropy_coef = .01,
    value_coef = .5,
    max_grad_norm = 1
)


q_config = dict(
    num_steps = 256,
    num_envs = 32,
    replacement_steps = 20,
    max_grad_norm = 1,
    eps = .8,
    observation_dim = 13 + 4,
    action_space_dim = 3,
    mini_batch_size = 128,
    update_epochs = 4,
    lam = .9
)


train_days = ROLLING_WINDOW_SIZE_IN_DAYS + MAX_TRADE_TIME_IN_DAYS*30
test_days = MAX_TRADE_TIME_IN_DAYS




if __name__ == "__main__":
    parser = argparse.ArgumentParser("training")

    parser.add_argument("-fp", "--filepath", required=True,
        help="path to stock file")
    parser.add_argument("-algo", "--algo", required=False, default= "PPO",
        help="Algo either PPO or Q_learning")
    
    parser.add_argument("-model_save_path", "--model_save_path", required= False, default= "models")
    parser.add_argument("-iterations", "--iterations", required= False, default= 4000)

    args = vars(parser.parse_args())
    

    df = pd.read_csv(args["filepath"])
    #df = pd.read_csv("data/5minute/HDFCBANK.csv")
    df = interpolatorNotForDay(df= df, mode= "5minute")
    addPatterns(df)



    if args["algo"] == "PPO":
        rolling_trainer = RollingTrain(df= df, train_days= train_days, test_days= test_days, ppo_config= ppo_config, device= device, save_dir= args["model_save_path"],
                     mode= "5minute", updates = int(args["iterations"])
                    )

    else:
        rolling_trainer = QRollingTrain(df= df,train_days= train_days,test_days= test_days, device= device,save_dir= args["model_save_path"],
                                        updates= int(args["iterations"]), q_config= q_config, mode= "5minute"
        )

    


    rolling_trainer.getRolling()
