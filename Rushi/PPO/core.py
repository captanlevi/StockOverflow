from .constants import MAX_TRADE_TIME_IN_DAYS, MAX_TRADE_TIME_IN_MINUTE, MAX_TRADE_TIME_IN_5_MINUTE, MODES, MAX_HOLDING, minutes_5_in_day, minutes_in_day,start_time,end_time_5_minute, end_time_minute
from evaluation.constants import HOLD_ACTION,BUY_ACTION,SELL_ACTION
from typing import Dict, List
from .utils import makeFeatures
import numpy as np
from evaluation.core import State
from .utils import getInvalidActionMask, getStateFeatures, getStockFeatures
import datetime
import pandas as pd
import random

class EarlyEnvironment:
    def __init__(self,df : Dict,mode : str, commision = .1, linspace = [1,1]):
        """
        linspace places sample importance, [1,1] is just equal importance
        """
        assert mode in MODES, "{} is invalid mode".format(mode)
        self.mode = mode
        self.data = makeFeatures(df = df, mode= mode)
        self.commision = commision
        self.max_holding = MAX_HOLDING
        self.max_length = None
        if mode == "minute":
            self.max_length = MAX_TRADE_TIME_IN_MINUTE
        elif mode == "day":
            self.max_length = MAX_TRADE_TIME_IN_DAYS
        else:
            self.max_length = MAX_TRADE_TIME_IN_5_MINUTE

        self.curr_iterator = None 
        self.num_episodes = 0
        self.info = dict(
            reward = 0,
            profit = 0,
            length = 0,
            trades = 0
        )
        self.possible_indices = self._getPosibleStartIndices()
        self.sample_weights = EarlyEnvironment.createWeights(l= len(self.possible_indices), linspace= linspace)
        self.reward_div = None


    @staticmethod
    def createWeights(l,linspace = [1,4]):
        weights = np.linspace(start= linspace[0], stop= linspace[1], num= l)
        weights = weights/weights.sum()
        return weights


    def _getPosibleStartIndices(self):
        interval = None
        if self.mode == "minute":
            interval = minutes_in_day
        else:
            interval = minutes_5_in_day

        max_index = len(self.data["close"]) - self.max_length
        possible_indices = []
        for i in range(0,len(self.data["timestep"]),interval):
            if i > max_index:
                break
            
            assert pd.to_datetime(self.data["timestep"][i]).time() == start_time
            possible_indices.append(i)
        
        return possible_indices



    def getStateFeatures(self):
        state = self.curr_iterator["state"]
        current_closing_price = self.data["close"][self.curr_iterator["index"]]
        return getStateFeatures(state= state, current_closing_price= current_closing_price, max_holding= self.max_holding)


    def returnObs(self):
        stock_features = getStockFeatures(length= self.curr_iterator["length"], features= self.data["features"][self.curr_iterator["index"]],
                                          max_length= self.max_length
                                        )
        state_features = self.getStateFeatures()
        invalid_action_mask = getInvalidActionMask(prev_action= self.curr_iterator["prev_action"],state= self.curr_iterator["state"],
                                                   max_holding= self.max_holding, length= self.curr_iterator["length"],
                                                   max_length= self.max_length, current_price= self.data["close"][self.curr_iterator["index"]]
                                                   )
        self.curr_iterator["prev_invalid_action_mask"] = invalid_action_mask
        return dict(next_obs = np.array(stock_features + state_features) , invalid_action_mask = invalid_action_mask)
    

    def reset(self):
        index = np.random.choice(a= self.possible_indices,size= 1, replace= False, p= self.sample_weights).item()
        self.curr_iterator = dict(index = index,reward = 0, length = 1, prev_action = HOLD_ACTION, profit = 0,trades = 0,
                                state = State(commision= self.commision,short= False))

        next_obs = self.returnObs()
        operating_data = self.data["close"][index:index+ self.max_length]
        self.reward_div = operating_data.max() - operating_data.min() - (operating_data.max() + operating_data.min())*self.commision/100
        #self.reward_div = operating_data.max()
        return dict(next_obs = next_obs["next_obs"], invalid_action_mask = next_obs["invalid_action_mask"])
    
    def step(self,action : int):
        assert action in [BUY_ACTION,SELL_ACTION,HOLD_ACTION], "{} is invalid action".format(action)
        assert self.curr_iterator["prev_invalid_action_mask"][action] == False, "the action in step is invalid"

        current_closing_price =  self.data["close"][self.curr_iterator["index"]]
        done = False
        if self.curr_iterator["length"] == self.max_length and action == HOLD_ACTION:
            end_time = None
            if self.mode == "minute":
                end_time = end_time_minute
            else:
                end_time = end_time_5_minute
            assert pd.to_datetime(self.data["timestep"][self.curr_iterator["index"]]).time() == end_time 
            done = True
        

        reward = 0
        if action == SELL_ACTION:
            state = self.curr_iterator["state"]
            mean_holding = sum(state.values)/len(state.values)
            reward = (current_closing_price - mean_holding - self.commision/100*(current_closing_price + mean_holding))/self.reward_div
            #reward = (current_closing_price - self.commision*current_closing_price/100)/self.reward_div
        #elif action == BUY_ACTION:
        #    reward = -(current_closing_price + self.commision*current_closing_price/100)/self.reward_div
            
        self.curr_iterator["profit"] += self.curr_iterator["state"].update(closing_price= current_closing_price, action= action)[0]
        # profit is different from reward, its the absolute value
        # state has been updated
        self.curr_iterator["reward"] += reward

        if action == HOLD_ACTION:
            self.curr_iterator["index"] += 1
            self.curr_iterator["prev_action"] = HOLD_ACTION
            self.curr_iterator["length"] += 1
        elif action == BUY_ACTION:
            self.curr_iterator["prev_action"] = BUY_ACTION
        else:
            self.curr_iterator["prev_action"] = SELL_ACTION
            self.curr_iterator["trades"] += 1

        # after we set the prev_action we can calculate the invalid_action_mask, as this mask applies on the next action
        if done == False:
            next_obs = self.returnObs()
            result  = dict(next_obs = next_obs["next_obs"], done = int(done), reward  = reward, invalid_action_mask = next_obs["invalid_action_mask"])
        else:
            self.num_episodes += 1
            self.info["profit"] = (self.info["profit"]*(self.num_episodes - 1) +  self.curr_iterator["profit"])/self.num_episodes
            self.info["reward"] = (self.info["reward"]*(self.num_episodes - 1) +  self.curr_iterator["reward"])/self.num_episodes
            self.info["length"] = (self.info["length"]*(self.num_episodes - 1) +  self.curr_iterator["length"] - 1)/self.num_episodes
            self.info["trades"] = (self.info["trades"]*(self.num_episodes - 1) +  self.curr_iterator["trades"])/self.num_episodes
            next_obs = self.reset()
            result  = dict(next_obs = next_obs["next_obs"], done = int(done), reward  = reward, invalid_action_mask = next_obs["invalid_action_mask"])

        return result





class EarlyEnvs:
    def __init__(self, early_environments : List[EarlyEnvironment]):
        self.early_environments = early_environments
    
    def returnObs(self):
        obs = list(map(lambda x : x.returnObs(), self.early_environments))
        next_obs = list(map(lambda x : x["next_obs"], obs))
        invalid_action_mask = list(map(lambda x : x["invalid_action_mask"], obs))
        return dict(
            next_obs = np.array(next_obs),
            invalid_action_mask = np.array(invalid_action_mask)
        )

    def reset(self):
        for early_environment in self.early_environments:
            early_environment.reset()
        return self.returnObs()
    
    def step(self,actions):
        assert len(actions) == len(self.early_environments)
        agg_results = []

        for i in range(len(actions)):
            agg_results.append(self.early_environments[i].step(action=actions[i]))
        
        keys = list(agg_results[0].keys())
        master_agg = dict()
        for key in keys:
            master_agg[key] = []
            for agg_result in agg_results:
                master_agg[key].append(agg_result[key])
            master_agg[key] = np.array(master_agg[key])
        
        return master_agg
    
    def getInfo(self):
        reward, length, profit, trades = 0,0,0,0
        for env in self.early_environments:
            reward += env.info["reward"]
            length += env.info["length"]
            profit += env.info["profit"]
            trades += env.info["trades"]
        
        return dict(
            reward = reward/len(self.early_environments),
            length = length/len(self.early_environments),
            profit = profit/len(self.early_environments),
            trades = trades/len(self.early_environments)
        )