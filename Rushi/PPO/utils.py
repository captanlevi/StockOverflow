from .constants import *
import numpy as np
from evaluation.constants import HOLD_ACTION,BUY_ACTION,SELL_ACTION
from evaluation.core import State
import pandas as pd
import pickle
from patternpy.tradingpatterns.tradingpatterns import *


class Logger:
    def __init__(self,name = "",verbose = True,**kwargs):
        """
        kwargs is a a dict mapping name of metric with an empty list or some values in case or resuming

        Logger populates each with [value1,value2, ....]
        """
        self.name = name
        self.verbose = verbose
        self.metrices_dict = kwargs
        self.metric_report_step_size = dict()
        self.default_step_size = 100

    def addMetric(self,metric_name,value):
        if metric_name not in self.metrices_dict:
            self.metrices_dict[metric_name] = []
        self.metrices_dict[metric_name].append(value)
        if self.verbose:
            self.reporting(metric_name= metric_name)
       
    def setMetricReportSteps(self,metric_name,step_size):
        self.metric_report_step_size[metric_name] = step_size

    
    def getMetric(self,metric_name):
        return self.metrices_dict[metric_name]
    
    def getAllMetricNames(self):
        return list(self.metrices_dict.keys())
    

    def save(self,path):
        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    

    @staticmethod
    def load(path):
        with open(path, 'rb') as inp:
            logger = pickle.load(inp)
        return logger



    def reporting(self,metric_name):
        """
        reports the mean of the last step_size of the metric
        if the the length%step_size is 0
        """
        step_size = self.metric_report_step_size[metric_name] if metric_name in self.metric_report_step_size else self.default_step_size

        if len(self.metrices_dict[metric_name]) % step_size == 0:
            values = self.metrices_dict[metric_name][-step_size:]
            print("{} ---- {} metric {} = {}".format(self.name,len(self.metrices_dict[metric_name]),metric_name,sum(values)/len(values)))



def makeFeatures(df, mode, add_patterns = True):
    """
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00333-6/tables/2  for more features
    returns a dict of features of shape (TS,feature_size) and close which is an 1-D array

    I am using window_size instead of window_size -1 so that the features start from the start of a day
    """
    def normalizeSeries(series : pd.Series, window_size):
        rolling_mean = series.rolling(window = window_size).mean()
        rolling_std = series.rolling(window = window_size).std()
        normalized = (series - rolling_mean)/(rolling_std + 1e-6)
        normalized = normalized.values[window_size:]
        normalized = normalized.clip(min = -5, max = 5)
        return normalized.reshape(-1,1)
    def getVol(series : pd.Series,window_size):
        log_returns = np.log(series.values[1:]/series.values[:-1])
        log_returns = pd.Series([0] + log_returns.tolist())
        vol = log_returns.rolling(window= window_size).std()*100
        vol = vol.values[window_size:]
        vol = vol.clip(min = -5, max = 5)
        return vol.reshape(-1,1)
    
    def getSlope(column_name,window_size):
        def calc_slope(x):
            slope = np.polyfit(range(len(x)), x, 1)[0]
            return slope
        
        slope = df.close.rolling(window= window_size).apply(calc_slope)
        return slope.values[window_size:].reshape(-1,1)
    
    def getSOcc(window_size):
        lowest = df.low.rolling(window= window_size).min()
        highest = df.high.rolling(window= window_size).max()
        s_oss = (df.close- lowest)/(highest - lowest)
        return s_oss.values[window_size:].reshape(-1,1)
        

    assert mode in MODES, "{} is not a valid mode".format(mode)
    window_size = None

    if mode == "day":
        window_size = ROLLING_WINDOW_SIZE_IN_DAYS
    elif mode == "5minute":
        window_size = ROLLING_WINDOW_SIZE_IN_5_MINUTE
    elif mode == "minute":
        window_size = ROLLING_WINDOW_SIZE_IN_MINUTE
    else:
        assert False, "mode {} is wrong".format(mode)

    norm_open = normalizeSeries(series= df.open, window_size= window_size)
    norm_high = normalizeSeries(series= df.high, window_size= window_size)
    norm_low  = normalizeSeries(series= df.low, window_size= window_size)
    norm_close = normalizeSeries(series= df.close, window_size= window_size)
    norm_vol  = normalizeSeries(series= df.volume, window_size= window_size)
    vol = getVol(series= df.close, window_size= window_size)
    s_oss = getSOcc(window_size= window_size)
    #slope = getSlope(column_name= "close", window_size= window_size)

    close = df["close"].values[window_size:]
    timestep = df["date"].values[window_size:]
    features = np.concatenate( [norm_open,norm_high, norm_low, norm_close,vol,s_oss,norm_vol], axis= 1)

    if add_patterns == True:
        pattern_columns = ["head_shoulder_pattern", "multiple_top_bottom_pattern",
                            "double_pattern", "pivot_signal"]
        patterns = df[pattern_columns].values[window_size:,:]
        features = np.concatenate([features,patterns], axis= 1)
    
    return dict(features = features, close = close, timestep = timestep)


def getInvalidActionMask(prev_action,state: State,max_holding : int, length : int, max_length : int,current_price : float):
    """
    length is the current timestep 
    max_length is the max timesteps the agent can oprate
    
    """
    invalid_action_mask = [False,False,False]
    if prev_action == BUY_ACTION:
        # cannot sell if bought, first it must hold
        invalid_action_mask[SELL_ACTION] = True

    elif prev_action == SELL_ACTION:
        # cannot buy if sold, first it must hold
        invalid_action_mask[BUY_ACTION] = True


    if len(state.values) == max_holding:
        # if the number of stocks in holding excede the max_holding                          
        invalid_action_mask[BUY_ACTION] = True
    
    if state.isHolding() == False:
        invalid_action_mask[SELL_ACTION] = True
    
    if length == max_length:
        # if the max_length  has been attained you can only sell all you have until you call hold
        if state.isHolding():
            invalid_action_mask[HOLD_ACTION] = True
        invalid_action_mask[BUY_ACTION] = True
    

    # stop loss
    """
    if len(state.values) != 0:
        mean_state_price = sum(state.values)/len(state.values)

        if (mean_state_price - current_price)/mean_state_price > .15:
            invalid_action_mask[BUY_ACTION] = True
            invalid_action_mask[HOLD_ACTION] = True            
    """

    
    assert np.array(invalid_action_mask).sum() < 3, "all actions are invalid"
    return invalid_action_mask



def getStateFeatures(state : State,current_closing_price,max_holding = MAX_HOLDING):
    features = [-1,-1]
    if state.isHolding():
        features[0] = len(state.values)/max_holding
        mean_holding = sum(state.values)/len(state.values)
        features[1] = (current_closing_price - mean_holding)/mean_holding/.05
        features[1] = min(max(-5, features[1]), 5)
    
    return features + features


def getStockFeatures(length,features,max_length):
    """
    Where length is the current timestep out of the max_length.
    Features is numpy array of current timestep.

    returns a 1-D list of features
    """
    return [length/max_length]*2 + features.tolist()



def interpolatorNotForDay(df : pd.DataFrame, mode):

    def getTimeRange(start,end,interval):
        time_range = [start]

        while True:
            if time_range[-1] >= end:
                break
            time_range.append(time_range[-1] + interval)
        return time_range


    start_time = datetime.time(hour= 9, minute= 15)
    end_time = None
    interval = None

    if mode == "minute":
        end_time = end_time_minute
        interval = datetime.timedelta(minutes= 1)
    elif mode == "5minute":
        end_time = end_time_5_minute
        interval = datetime.timedelta(minutes= 5)
        

    df["day"] = pd.to_datetime(df["date"]).apply(lambda x : x.date())
    values_interpolated = 0

    new_df = []
    for day,mini_df in df.groupby(by= "day"):
        if day.strftime("%A") in ["Saturday", "Sunday"]:
            # in cases for weekends
            continue
        mini_df.drop(columns= ["day"], inplace= True)
        start_date_time =  datetime.datetime.combine(date= day, time= start_time)
        end_date_time = datetime.datetime.combine(date= day, time= end_time)
        time_range =  getTimeRange(start= start_date_time, end= end_date_time, interval= interval)

        new_mini_df = pd.DataFrame(data= dict(date = time_range))
        mini_df["date"] = mini_df.date.apply(lambda x : str(x))
        new_mini_df["date"] = new_mini_df.date.apply(lambda x : str(x))
        #print(mini_df)
        
        new_mini_df = new_mini_df.merge(mini_df, on= "date",how= "left")
        #for i in range(len(new_mini_df)):
        values_interpolated += new_mini_df.isna().sum().sum()

        new_mini_df["date"] = pd.to_datetime(new_mini_df.date)

        
        new_mini_df = new_mini_df.interpolate(method='linear', axis=0).ffill().bfill()

        if new_mini_df.isna().sum().sum() > 0:
            # in cases of public holidays
            continue
        new_df.append(new_mini_df)
    

    new_df = pd.concat(new_df)
    new_df.reset_index(drop=True, inplace=True)
    print("total values interpolated = {}".format(values_interpolated))
    new_df.sort_values(by= "date", inplace= True)
    new_df["date"] = new_df["date"].apply(lambda x : str(x))
    return new_df
        



def addPatterns(df : pd.DataFrame):
    def adjustForPeek(column_name):
        df[column_name] = df[column_name].shift(-1)
        df.loc[len(df) -1, column_name] = 0

    def revertDfForTrading(df : pd.DataFrame):
        keep_columns = ["date","open", "high", "low", "close", "volume", "head_shoulder_pattern", "multiple_top_bottom_pattern",
                        "double_pattern", "pivot_signal", "triangle_pattern"
                        ]
        total_columns = list(df.columns)
        remove_columns = list(filter(lambda x : x not in keep_columns, total_columns))
        df.drop(columns= remove_columns, inplace= True)

    detect_head_shoulder(df= df)
    adjustForPeek(column_name= "head_shoulder_pattern")
    detect_multiple_tops_bottoms(df= df)
    detect_double_top_bottom(df= df)
    adjustForPeek(column_name= "double_pattern")

    detect_triangle_pattern(df= df)
    find_pivots(df= df)
    df.rename(columns= dict(signal = "pivot_signal"), inplace= True)
    adjustForPeek(column_name= "pivot_signal")

    revertDfForTrading(df= df)