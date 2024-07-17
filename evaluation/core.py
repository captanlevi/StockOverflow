from .constants import *

class State:
    def __init__(self,commision : float, short : bool = False):
        """
        values store your bought stocks 
        it is sorted in reverse order so the cheapest price is at the last 

        commision - commision in percentage
        """
        self.values = []
        self.commision = commision
        self.short = short

    def calcCommision(self,value):
        return value*self.commision/100
    

    def update(self,closing_price : float,action : int):
        """
        When action is buy, check for any shorted stocks and square it
        when action is sell, get the cheapest stock price bought from values and return profit
        If you already have stocks bought YOU CANNOT SHORT
        If you have no stocks and action is sell and is_short is True you short stock

        returns (money,trade_completed)
        money -  the amount of money spent/made on executing action (buy,hold or sell) on this state. -ve if you get money +ve if you give money
                 this amount 
        trade_completed - True if either a buy-sell or sell-buy cycle is complete else False


        """
        if action not in [BUY_ACTION,SELL_ACTION,HOLD_ACTION]:
            raise ValueError("action should be in buy sell or hold defined in constants")
        
  
        if action == HOLD_ACTION:
            return 0,False
        elif action == BUY_ACTION:
            if len(self.values) >= 1 and self.values[-1] < 0:
                # squaring off shorted stock
                self.values.pop()
                return -(closing_price + self.calcCommision(closing_price)), True
            
            self.values.append(closing_price)
            self.values.sort(reverse= True)  # keep the values sorted in reverse order
            return -(closing_price + self.calcCommision(closing_price)), False
        else:
            # this is the selling case
            # in this case we get the min value from values and return it
            if len(self.values) >= 1:
                self.values.pop() # removing the cheapest price
                return closing_price -self.calcCommision(closing_price),True
            else:
                assert self.short == True, "cant sell stocks when shorting is not allowed and you are broke (have no stocks)"
                # will append the negative of the closing price
                self.values.append(-closing_price),False
                self.values.sort()
                return closing_price -self.calcCommision(closing_price), False
            
    def isHolding(self):
        if len(self.values) == 0:
            return False
        if self.values[0] > 0:
            # if the largest price bought is not shorted then we are holding
            return True
        
        # In this case all values are negative ie all stocks are shorted (so we are not holding any)
        return False

    def isShorted(self):
        if len(self.values) == 0:
            return False
        return self.values[-1] < 0

    def getStockValue(self, closing_price):
        if self.isHolding() == False:
            return 0
        return len(self.values)*(closing_price - self.calcCommision(closing_price))


class BaseStrategy:
    """
    This is the base strategy, other methods can be implemented to work with the step method
    """
    def step(self,data,state : State,current_pointer : int):
        """
        data is pandas Dataframe with columns data,open.high,low,close,volume
        current_pointer is the index of the data the strategy must act on
        state is the current state of your portfolio, (please read the class defination)

        return value is one of {HOLD_ACTION, BUY_ACTION, SELL_ACTION} which are defined in constants
        This method will be called from the evaluator,
        the current_pointer is guaranteed to increase in step of 1, when called from evaluator in single model
        the current_pointer will keep repeating when evaluator is in multi mode untill the startegy outputs HOLD
        

        data at current_pointer can be accesses by 
        row = data.iloc[current_pointer]

        date,open,high,low,close,volume = row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]

        ***************************************************
        PLEASE DO NOT MODIFY data INPLACE !!!!!!!!!
        ***************************************************
        """

        raise NotImplementedError()