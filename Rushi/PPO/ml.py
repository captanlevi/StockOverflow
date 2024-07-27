import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CategoricalMasked(Categorical):
    def __init__(self,device ,probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        self.device = device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)
    



class PPOAgent(nn.Module):
    """
    Can experiment with bidirectional, it will get complicated tho
    """
    def __init__(self, lstm_input_size, lstm_hidden_size, output_dim, layers=1) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size= lstm_input_size, hidden_size= lstm_hidden_size, num_layers= layers,bidirectional= False,
                            batch_first= True
                            )

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.action_linear = nn.Sequential(nn.Linear(lstm_hidden_size, lstm_hidden_size//2), nn.Tanh(),
                                           nn.Linear(lstm_hidden_size//2, lstm_hidden_size//2), nn.Tanh(),
                                           nn.Linear(lstm_hidden_size//2, output_dim)
                                            )
        self.value_linear = nn.Sequential(nn.Linear(lstm_hidden_size, lstm_hidden_size//2), nn.Tanh(),
                                           nn.Linear(lstm_hidden_size//2, lstm_hidden_size//2), nn.Tanh(),
                                           nn.Linear(lstm_hidden_size//2, 1)
                                            )

    def getStates(self,x,lstm_state,done):
        """
        x is (seq,feature_dim) where seq is seq_len*BS
        done is (seq,BS)
        seq_len is 1 when I am rolling out
        where BS is the number of environments

        In case of inference control the batch size by using the lstm_state size (all zeros but with different size)
        
        """
        batch_size = lstm_state[0].shape[1]

        x = x.reshape(-1,batch_size,self.lstm.input_size).permute((1,0,2))
        done = done.reshape(-1,batch_size).permute((1,0))   # for the case where 
        output = []
        seq_len = x.shape[1]

        for i in range(seq_len):

            lstm_hidden,lstm_state = self.lstm(x[:,i:i+1,:],
                (
                    (1.0 - done[:,i]).view(1, -1, 1) * lstm_state[0],
                    (1.0 - done[:,i]).view(1, -1, 1) * lstm_state[1],
                )

            # lstm_hidden is (BS,1,lstm_output_size)
            )

            output.append(lstm_hidden.permute((1,0,2)))
        
        output_hidden = torch.flatten(torch.cat(output),0,1)
        return output_hidden,lstm_state


    def getValue(self, x, lstm_state, done):
        hidden, _ = self.getStates(x, lstm_state, done)
        return self.value_linear(hidden)

    def getActionAndValue(self, x, invalid_action_mask,lstm_state, done, action=None):
        """
        x is (BS,obs_dim which is feature dim for each timestep in this case)
        invalid_action_mask is (BS,3)
        """
        hidden, lstm_state = self.getStates(x, lstm_state, done)
        logits = self.action_linear(hidden)
        probs = CategoricalMasked(device= x.device,logits= logits, masks= ~invalid_action_mask)
        # with the categoricalMasked we will only sample valid actions

        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.value_linear(hidden), lstm_state
    

    def evalForward(self,x,invalid_action_mask,lstm_state):
        """
        Here x is of shape (BS,feature_dim)
        invalid_action_masks is of shape (BS,num_actions)
        returns the action probablity distrubation for this timestep
        """
        hidden,lstm_state = self.lstm(x.unsqueeze(1), lstm_state)
        hidden = hidden[:,0,:]
        logits = self.action_linear(hidden)
        probs = CategoricalMasked(logits= logits, masks= ~invalid_action_mask, device= x.device)
        return probs,lstm_state
    
    

    def save(self,path):
        torch.save(self.state_dict(),path)
    
    def load(self,path):
        self.load_state_dict(torch.load(path))
        