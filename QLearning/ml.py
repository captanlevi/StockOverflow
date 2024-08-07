import torch.nn as nn
import torch
from Rushi.PPO.ml import CategoricalMasked
import numpy as np

class DuelingQNetwork(nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super().__init__()
        self.value_linear = nn.Sequential(nn.Linear(input_dim,input_dim*2), nn.ReLU(),
                                           nn.Linear(input_dim*2, input_dim*4), nn.ReLU(),
                                           nn.Linear(input_dim*4, 1)
                                           )
        self.advantage_linear = nn.Sequential(nn.Linear(input_dim,input_dim*2), nn.ReLU(),
                                           nn.Linear(input_dim*2, input_dim*4), nn.ReLU(),
                                           nn.Linear(input_dim*4, output_dim)
                                    )

    def forward(self,X):
        values = self.value_linear(X)
        advantage = self.advantage_linear(X)
        q_values = advantage - torch.mean(advantage,dim= -1, keepdim= True) + values
        return q_values
    

    def getAction(self,X, invalid_action_mask, eps = None):
        """
        X is (BS,obs_dim)
        """
        q_values = self.forward(X)
      

        if eps == None or np.random.random() > eps:
            q_values[invalid_action_mask] = -10000000
            chosen_values, chosen_actions = torch.max(q_values,dim = -1) 

            return chosen_values,chosen_actions        
        else:

            sample_probs = torch.ones_like(invalid_action_mask).float().to(X.device)
            sample_probs = CategoricalMasked(logits= sample_probs, masks= ~invalid_action_mask, device= X.device)
            chosen_actions = sample_probs.sample()
            chosen_values = torch.gather(input= q_values, dim= -1, index = chosen_actions.view(-1,1))[:,0]
            
            return chosen_values, chosen_actions


    def save(self,path):
        torch.save(self.state_dict(),path)
    
    def load(self,path):
        self.load_state_dict(torch.load(path))
        
