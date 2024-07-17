import pandas as pd
from .core import *
from .constants import *
from .exceptions import *
import matplotlib.pyplot as plt


class BackTest:
    """
    BackTest takes in a startegy and runs it on the provided data.
    """
    def __init__(self,data : pd.DataFrame, strategy : BaseStrategy,initial_cash = 10000,commision = .1, mode = "single",short = False):
        """
        data (pd.Dataframe)     - Columns date,open,high,low,close
        strategy (BaseStrategy) - Custom strategy that must have BaseStrategy as super class
        initial_cash (float)    - Money you start with
        commision (float)       - Commision you pay for executing a buy or sell ( in percentage)
        mode (str)              - Mode is either single or multi 
        short (bool)            - Is shorting allowed


        When in single mode the strategy cannot hold more than one stock at a time, when in multi it can.
        When short is True, shorting a stock is allowed (please be careful as positions have to be squared within the day, and this logic has to be handled within the strategy)
        Recommended to set short to false

        ***************
        All buys and sells will be executed using the closing price of the current_pointer
        ***************
        """
        self.data = data
        self.current_pointer = 0
        self.strategy = strategy
        self.state : State = State(commision= commision, short= short)
        self.short = short
        self.mode = mode
        self.initial_cash = initial_cash
        self.cash = initial_cash

        self.current_date = None

        self.log = dict(
            timestamp = [],
            money = [],
            is_trade_complete = [],
            stock_value = [],
            total_value = []
        )

        self.step_trade_counter = 0
    def __step(self):
        if self.step_trade_counter > MAX_TRADES_PER_TIMESTAMP:
            raise MaxTradePerTimestepExceded()
        
        timestamp = pd.to_datetime(self.data.iloc[self.current_pointer]["date"])

        # the following logic enforces shorted stocks to be squared the same day
        if self.current_date == None:
            self.current_date = timestamp.date()
        else:
            if self.current_date != timestamp.date():
                if self.state.isShorted():
                    raise NotSquaredSameDay()
            self.current_date = timestamp.date()
                    


        closing_price = self.data.iloc[self.current_pointer]["close"]
        action = self.strategy.step(data= self.data,current_pointer= self.current_pointer,state= self.state)
        if action not in [BUY_ACTION,SELL_ACTION,HOLD_ACTION]:
            raise ValueError("action should be on of buy sell or hold defined in constants")
        
        is_holding = self.state.isHolding()
        money,is_trade_complete = self.state.update(closing_price= closing_price,action= action)
        if action == BUY_ACTION:
            assert money < 0
            if  is_holding and self.mode == "single":
                raise ValueError("cant buy more than one stock in single mode !!!!")
            
            if self.cash < -money:
                raise NoMoneyNoHoney
            self.cash += money
        elif action == SELL_ACTION:
            if is_holding == False and self.short == False:
                raise ValueError("cant sell stock while not holding any [set short to True first]")
            self.cash += money

        
        self.log["timestamp"].append(timestamp)
        self.log["money"].append(self.cash)
        self.log["is_trade_complete"].append(is_trade_complete)
        self.log["stock_value"].append(self.state.getStockValue(closing_price= closing_price))
        self.log["total_value"].append(self.cash +self.state.getStockValue(closing_price= closing_price))

        self.step_trade_counter += 1
        if self.mode == "multi":
            # if mode is multi keep stepping until the strategy gives out hold action
            self.step()
        
    
    def _step(self):
        """
        This method is created so that recursive calls in case of multi do not increment current_pointer 
        multiple times
        """
        self.step_trade_counter = 0
        self.__step()
        self.current_pointer += 1
        
    
    def runStrategy(self,limits = None):
        """
        limits : List
        limits is an optional argument that defines start and end indices for the DF ex [1200,1600]
        """

        start, end = 0, len(self.data)
        if limits is not None:
            start,end = limits
            assert start < end, "wrong trading limits {},{}".format(start,end)
        
        self.current_pointer = 0
        while self.current_pointer < end:
            try:
                self._step()
            except NoMoneyNoHoney:
                print("RAN out of money !!!!")
                break
        self.pprint()

    def pprint(self):
        print("Ran from {} to {}".format(self.log["timestamp"][0], self.log["timestamp"][-1]))
        print("Initial value = {}".format(self.initial_cash))
        print("Final value = {}".format(self.log["total_value"][-1]))

    def plotValue(self):
        plt.plot(self.log["total_value"])
        plt.show()

    

    
    