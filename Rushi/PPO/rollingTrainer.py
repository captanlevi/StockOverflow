from .constants import *
from .trainer import PPOTrainer
from .ppoStrategy import PPOAgent
from .core import EarlyEnvironment, EarlyEnvs
from .ppoStrategy import PPOStrategy
import pandas as pd
import os
from evaluation.backTest import BackTest
from .utils import Logger




class RollingTrain:
    def __init__(self,df,train_days, test_days, ppo_config,device,save_dir,mode,agent = None, updates = 1000):
        self.df = df
        self.train_days = train_days
        self.test_days = test_days
        self.ppo_config = ppo_config
        self.device = device
        self.save_dir = save_dir
        self.roll = 1
        if agent == None:
            self.agent = PPOAgent(lstm_hidden_size= 128,lstm_input_size= ppo_config["observation_dim"], layers= 2, output_dim= ppo_config["action_space_dim"])
        else:
            self.agent = agent
        
        self.agent.to(self.device)

        assert mode in MODES
        self.mode = mode
        unit = None
        if mode == "5minute":
            unit = minutes_5_in_day
        elif(mode == "minute"):
            unit = minutes_in_day
        else:
            unit = 1
        self.unit = unit

        self.train_indices = [0,self.train_days*unit -1]
        self.test_indices = [self.train_days*unit, self.train_days*unit + self.test_days*unit  - 1]
        self.updates = updates


        self.profit = 0
    def _rollIndices(self):
        self.roll += 1
        self.train_indices[0] += self.test_days*self.unit
        self.train_indices[1] += self.test_days*self.unit
        self.test_indices[0] += self.test_days*self.unit
        self.test_indices[1] += self.test_days*self.unit

    
    def getTrainEnvs(self, linspace = [1,1]):
        envs = []
        for _ in range(self.ppo_config["num_envs"]):
            envs.append(EarlyEnvironment(df = self.df.iloc[self.train_indices[0]:self.train_indices[1] + 1], mode= self.mode,linspace= linspace))
        early_envs = EarlyEnvs(early_environments= envs)
        return early_envs
    
    
    def evalRoll(self):
        ppo_strategy = PPOStrategy(df= self.df,mode= self.mode, agent= self.agent, device= self.device)
        back_test = BackTest(data= self.df, strategy= ppo_strategy, initial_cash= 100000, commision= .1, mode= "multi", short= False)
        limits = [self.test_indices[0],self.test_indices[1]]
        back_test.runStrategy(limits= limits)
        trades = sum(back_test.log["is_trade_complete"])
        print("num_trades =  {}".format(trades))
        return back_test.cash - back_test.initial_cash


    def save(self, logger : Logger = None):
        path = self.getAgentPath(roll= self.roll)
        self.agent.save(path= path)
        if logger == None:
            return
        logger_path = self.getLoggerPath(roll= self.roll)
        logger.save(path= logger_path)


    def getAgentPath(self,roll):
        return os.path.join(self.save_dir, "agent_roll_{}.pt".format(roll))
    def getLoggerPath(self,roll):
        return os.path.join(self.save_dir, "logger_roll_{}.pt".format(self.roll))

    
    def load(self,roll):
        path = self.getAgentPath(roll= roll)
        self.agent.load(path= path)
        logger_path = self.getLoggerPath(roll= roll)
        if os.path.exists(logger_path):
            return Logger.load(path= logger_path)
        return None

    def trainSingleRoll(self):
        logger = Logger(name= "STONK_ROLL_{}".format(self.roll))
        logger.default_step_size = 1000
        exists = False
        first_roll = self.roll == 1
        if os.path.exists(self.getAgentPath(roll= self.roll)):
            loaded_logger = self.load(roll= self.roll)
            if loaded_logger != None:
                logger = loaded_logger
            return None

        
 
        envs = self.getTrainEnvs(linspace= [1,1] if first_roll == True else [1,4])
        trainer = PPOTrainer(envs= envs, agent= self.agent, ppo_config= self.ppo_config, device= self.device, logger= logger)
        num_updates = self.updates if first_roll == True else 100
        print(num_updates)
        trainer.train(num_updates= num_updates, lr= 3e-4)
        return logger

    

    def getRolling(self):
        while self.test_indices[-1] < len(self.df):
            #self.load(roll= self.roll)
            logger = self.trainSingleRoll()
            self.profit += self.evalRoll()
            if logger != None:
                self.save(logger= logger)
            self._rollIndices()

            print("profit = {}".format(self.profit))
        


        
        

