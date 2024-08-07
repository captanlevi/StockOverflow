import torch
from copy import deepcopy
from Rushi.PPO.core import EarlyEnvironment, EarlyEnvs
from Rushi.PPO.utils import Logger
from .ml import DuelingQNetwork
import torch.nn as nn
import numpy as np



class QTrainer:
    def __init__(self,envs : EarlyEnvs,agent : DuelingQNetwork, device,logger : Logger, q_config : dict):
        self.envs = envs
        self.q_config = q_config
        self.agent = agent.to(device)
        self.lag_agent  = deepcopy(agent)
        self.device = device
        self.storage = []
        self.logger = logger
        
        self.best = dict(
            score = 0,
            model = None
        )
        self.logger.setMetricReportSteps(metric_name= "avg_reward", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_profit", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_length", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_trades", step_size= 1)

        self.storage = self._initializeStorage()
        self.logger = logger
        self.mse_loss_function = nn.MSELoss(reduction= "none")
        
    
    def _initializeStorage(self):
        config = self.q_config
        num_steps,num_envs,observation_dim, action_space_dim = config["num_steps"], config["num_envs"], config["observation_dim"], config["action_space_dim"]



        obs = torch.zeros([num_steps, num_envs,observation_dim]).to(self.device)
        invalid_action_masks = torch.zeros([num_steps,num_envs,action_space_dim], dtype= torch.bool).to(self.device)
        actions  = torch.ones([num_steps,num_envs]).to(self.device)*-1
        rewards = torch.zeros([num_steps,num_envs]).to(self.device)
        dones = torch.zeros([num_steps,num_envs]).to(self.device)


        return  dict(
            obs = obs, actions = actions,invalid_action_masks = invalid_action_masks,
            rewards = rewards, dones = dones
        )
    

    def runUpdate(self, agent_optimizer, prev_state, eps):
        num_steps,num_envs = self.q_config["num_steps"], self.q_config["num_envs"]
        


        next_obs = prev_state["next_obs"]
        next_invalid_action_masks = prev_state["next_invalid_action_masks"]
        next_done = prev_state["next_done"]


        for step in range(num_steps):
            self.storage["obs"][step] = next_obs
            self.storage["dones"][step] = next_done

            with torch.no_grad():
                q_value,action =  self.agent.getAction(X = next_obs,invalid_action_mask= next_invalid_action_masks, eps= eps)
                
                # put episode logging code here
                self.storage["actions"][step] = action
                env_step = self.envs.step(action.cpu().numpy())

                next_obs, reward, next_done, next_invalid_action_masks = env_step["next_obs"], env_step["reward"], env_step["done"], env_step["invalid_action_mask"]
                self.storage["rewards"][step] = torch.tensor(reward).to(self.device).float()
                

                next_obs, next_done = torch.tensor(next_obs).to(self.device).float(), torch.tensor(next_done).to(self.device)
                next_invalid_action_masks = torch.tensor(next_invalid_action_masks).to(self.device)
                


        prev_state = dict(
            next_obs = next_obs,
            next_done = next_done,
            next_invalid_action_masks = next_invalid_action_masks,
        )


        self._updateModel(agent_optimizer= agent_optimizer)

        return prev_state
        
            
            
    def _updateModel(self, agent_optimizer):
        self.agent.train()
        num_envs = self.q_config["num_envs"]
        num_steps = self.q_config["num_steps"]
        epochs = self.q_config["update_epochs"]
        batch_size = self.q_config["mini_batch_size"]
        lam = self.q_config["lam"]


        buffer = dict(state = [], next_state = [], actions = [], rewards = [], dones = [], next_invalid_action_masks = [])

        for i in range(num_steps -1):
            buffer["state"].append(self.storage["obs"][i])
            buffer["next_state"].append(self.storage["obs"][i+1])
            buffer["rewards"].append(self.storage["rewards"][i])
            buffer["actions"].append(self.storage["actions"][i])
            buffer["dones"].append(self.storage["dones"][i])
            buffer["next_invalid_action_masks"].append(self.storage["invalid_action_masks"][i+1])

        
       

        buffer["state"] = torch.concatenate( buffer["state"] ,dim= 0)
        buffer["next_state"] = torch.concatenate( buffer["next_state"] ,dim = 0)
        buffer["rewards"] = torch.concatenate( buffer["rewards"], dim= 0 ).reshape(-1,1)
        buffer["actions"] = torch.concatenate( buffer["actions"], dim= 0 ).reshape(-1,1).long()
        buffer["dones"] = torch.concatenate( buffer["dones"] , dim= 0).reshape(-1,1).bool()
        buffer["next_invalid_action_masks"] = torch.concatenate( buffer["next_invalid_action_masks"] , dim= 0)




        L = len(buffer["state"])
        for epoch in range(epochs):
            for _ in range(L//batch_size):

                indices = torch.tensor(np.random.randint(0,L,batch_size)).to(self.device)

                batch_states = buffer["state"][indices]
                batch_next_state = buffer["next_state"][indices]
                batch_rewards = buffer["rewards"][indices]
                batch_dones = buffer["dones"][indices]
                batch_actions = buffer["actions"][indices]
                batch_next_invalid_action_masks = buffer["next_invalid_action_masks"][indices]

                predicted_values = self.agent(batch_states)
                predicted_values_for_taken_action = torch.gather(input= predicted_values, dim= 1,index= batch_actions)[:,0] # (BS)

                
                with torch.no_grad():
                    _,next_state_max_actions_model = self.agent.getAction(batch_next_state,batch_next_invalid_action_masks)
                    next_state_values_lag_model = self.lag_agent(batch_next_state)
                    next_state_values_for_max_action = torch.gather(input= next_state_values_lag_model, dim= 1, index= next_state_max_actions_model.reshape(-1,1))[:,0] # (BS)
                    next_state_values_for_max_action = next_state_values_for_max_action*(~(batch_dones[:,0]))
                    target = batch_rewards[:,0] + lam*(next_state_values_for_max_action.squeeze()) # (BS)
                
                q_loss = (torch.abs(target - predicted_values_for_taken_action)).mean()
                loss = q_loss
                agent_optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.agent.parameters(), self.q_config["max_grad_norm"])
                agent_optimizer.step()
                self.logger.addMetric(metric_name= "q_loss", value= q_loss.item())


    

    def train(self, num_updates = 10, lr = 3e-4):
        
        agent_optimizer = torch.optim.Adam(params= self.agent.parameters(), lr= lr)

        num_steps,num_envs = self.q_config["num_steps"], self.q_config["num_envs"]
        eps = self.q_config["eps"]
        reset = self.envs.reset()

        next_obs = torch.tensor(reset["next_obs"]).to(self.device).float()
        next_invalid_action_masks = torch.tensor(reset["invalid_action_mask"]).to(self.device)
        next_done = torch.zeros(self.q_config["num_envs"]).to(self.device)

        prev_state = dict(
            next_obs = next_obs,
            next_done = next_done,
            next_invalid_action_masks = next_invalid_action_masks
        )


        for update in range(num_updates):
            eps = eps*.993
            eps = max(.1,eps)
            self.storage = self._initializeStorage()
            prev_state = self.runUpdate(agent_optimizer = agent_optimizer, prev_state= prev_state, eps = eps)
            if update%100 == 0:
                info = self.envs.getInfo()

                self.logger.addMetric(metric_name= "avg_reward", value= info["reward"])
                self.logger.addMetric(metric_name= "avg_length", value= info["length"])
                self.logger.addMetric(metric_name= "avg_profit", value= info["profit"])
                self.logger.addMetric(metric_name= "avg_trades", value= info["trades"])

            
            if update%self.q_config["replacement_steps"] == 0:
                print("replacing model")
                self.lag_agent = deepcopy(self.agent)
                self.lag_agent.eval()
            
            
            