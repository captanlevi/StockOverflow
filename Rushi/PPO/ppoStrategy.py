from evaluation.core import State
from evaluation.constants import *
from evaluation.core import BaseStrategy
from .constants import *
from .ml import PPOAgent
from .utils import *
import torch


class PPOStrategy(BaseStrategy):
    def __init__(self,df,mode,agent : PPOAgent, device):
        assert mode in MODES, "{} is invalid mode".format(mode)
        self.df = df
        self.mode = mode
        self.agent = agent
        self.agent.eval()
        self.features = makeFeatures(df= df, mode= mode)
        self.first_start_index = len(self.df) - len(self.features["close"])

        self.prev_action = HOLD_ACTION
        self.max_holding = MAX_HOLDING
        self.max_length = None
        self.device = device

        self.max_length = None
        if mode == "minute":
            self.max_length = MAX_TRADE_TIME_IN_MINUTE
        elif mode == "day":
            self.max_length = MAX_TRADE_TIME_IN_DAYS
        else:
            self.max_length = MAX_TRADE_TIME_IN_5_MINUTE
        
        self.current_length = 1
        self.lstm_state = (
        torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device),
        torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device),
        )
        
    def step(self, data, state: State, current_pointer: int):
        # got to reset everytime length excedes max_length and assert that we are not holding any stocks in state
        if current_pointer < self.first_start_index:
            return HOLD_ACTION
        assert self.df.iloc[current_pointer]["close"] == self.features["close"][current_pointer - self.first_start_index]
        
        current_price = self.df.iloc[current_pointer]["close"]

        stock_features = getStockFeatures(length= self.current_length, features= self.features["features"][current_pointer - self.first_start_index],max_length= self.max_length)
        state_features = getStateFeatures(state= state,current_closing_price= self.df.iloc[current_pointer]["close"])

        feature = np.array(stock_features + state_features)
        invalid_action_mask = getInvalidActionMask(prev_action= self.prev_action, state= state, max_holding= self.max_holding, max_length= self.max_length,
                                                   length= self.current_length, current_price= current_price
                                                   )


    

        feature = torch.tensor(feature).float().to(self.device).reshape(1,-1)
        invalid_action_mask = torch.tensor(invalid_action_mask).to(self.device).reshape(1,-1)


        with torch.no_grad():
            action_dist, self.lstm_state = self.agent.evalForward(x= feature, invalid_action_mask= invalid_action_mask,lstm_state= self.lstm_state)
            # action_dist is (1,3)
            action_dist = action_dist

        action = torch.argmax(action_dist.probs[0]).cpu().item()
        #action = action_dist.sample()[0].cpu().item()

        # set the prev action here
        self.prev_action = action


        if action == HOLD_ACTION:
            if self.current_length == self.max_length:
                assert len(state.values) == 0
                end_time = None
                if self.mode == "minute":
                    end_time = end_time_minute
                else:
                    end_time = end_time_5_minute
                assert pd.to_datetime(data.iloc[current_pointer].date).time() == end_time, "{}".format(data.iloc[current_pointer].date)
                self.current_length = 1
                self.lstm_state = (
                            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device),
                            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device),
                            )
            else:
                self.current_length += 1

        return action
