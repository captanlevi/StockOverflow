import torch
from .core import *
from .ml import PPOAgent
from .utils import Logger
import numpy as np
import torch.nn as nn


class PPOTrainer:
    def __init__(self,envs : EarlyEnvs,agent : PPOAgent,ppo_config : dict, device,logger : Logger):
        self.envs = envs
        self.ppo_config = ppo_config
        self.agent = agent.to(device)
        self.device = device
        assert len(self.envs.early_environments) == ppo_config["num_envs"]
        self.storage = self._initializeStorage()
        self.logger = logger
        
        self.best = dict(
            score = 0,
            model = None
        )
        self.logger.setMetricReportSteps(metric_name= "avg_reward", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_profit", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_length", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "avg_trades", step_size= 1)
        
    def _initializeStorage(self):
        config = self.ppo_config
        num_steps,num_envs,observation_dim, action_space_dim = config["num_steps"], config["num_envs"], config["observation_dim"], config["action_space_dim"]

        obs = torch.zeros([num_steps, num_envs,observation_dim]).to(self.device)
        invalid_action_masks = torch.zeros([num_steps,num_envs,action_space_dim], dtype= torch.bool).to(self.device)
        actions  = torch.ones([num_steps,num_envs]).to(self.device)*-1
        log_probs = torch.zeros([num_steps,num_envs]).to(self.device)
        rewards = torch.zeros([num_steps,num_envs]).to(self.device)
        dones = torch.zeros([num_steps,num_envs]).to(self.device)
        values = torch.zeros([num_steps,num_envs]).to(self.device)

        return  dict(
            obs = obs, actions = actions, log_probs = log_probs,invalid_action_masks = invalid_action_masks,
            rewards = rewards, dones = dones,values = values
        )


    def _calcAdvantage(self,next_obs,next_lstm_state,next_done):
        num_steps = self.ppo_config["num_steps"]
        rewards = self.storage["rewards"]
        with torch.no_grad():
            # originally next_value is (num_envs,1) dimentional
            next_value = self.agent.getValue(
                    next_obs,
                    next_lstm_state,
                    next_done,
                )[:,0]
        
        returns = torch.zeros_like(rewards).to(self.device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1 - next_done
                next_return = next_value
            else:
                next_non_terminal = 1 - self.storage["dones"][t+1] # t+1 as the does are shifted ( hence next done)
                next_return = returns[t+1]
            returns[t] = rewards[t] + self.ppo_config["gamma"]*next_non_terminal*next_return
        
        return returns , returns - self.storage["values"]
            
            
    def _updateModel(self, agent_optimizer):
        self.agent.train()
        num_envs, num_mini_batches = self.ppo_config["num_envs"], self.ppo_config["num_mini_batches"]
        num_steps = self.ppo_config["num_steps"]
        assert num_envs%num_mini_batches == 0

        # by reshaping -1 we get [t1,t1,t1,t1... num_envs times, t2,t2,t2....num_envs times]
        b_obs = self.storage["obs"].reshape(-1, self.ppo_config["observation_dim"])
        b_log_probs = self.storage["log_probs"].reshape(-1)
        b_actions = self.storage["actions"].reshape(-1)
        b_dones = self.storage["dones"].reshape(-1)
        b_advantages = self.storage["advantages"].reshape(-1)
        b_returns = self.storage["returns"].reshape(-1)
        b_values = self.storage["values"].reshape(-1)
        b_invalid_action_masks  = self.storage["invalid_action_masks"].reshape(-1, self.ppo_config["action_space_dim"])

        envs_per_batch = num_envs//num_mini_batches
        env_indices = np.arange(num_envs) # (0,1,2,3) for nenvs = 4
        flat_indices = np.arange(num_envs*num_steps).reshape(num_steps,num_envs)  # [(0,1,2,3),(4,5,6,7), .....] for nenvs = 4

        for epoch in range(self.ppo_config["update_epochs"]):
            np.random.shuffle(env_indices)

            for start in range(0,num_envs,envs_per_batch):
                end = start + envs_per_batch
                mb_env_indices = env_indices[start : end]
                mb_indices = flat_indices[:,mb_env_indices].ravel()
                _, new_log_prob, entropy, newvalue, _ = self.agent.getActionAndValue(
                    x = b_obs[mb_indices], invalid_action_mask= b_invalid_action_masks[mb_indices],
                    lstm_state= (
                        torch.zeros(self.agent.lstm.num_layers,envs_per_batch, self.agent.lstm.hidden_size).to(self.device),
                        torch.zeros(self.agent.lstm.num_layers,envs_per_batch, self.agent.lstm.hidden_size).to(self.device)
                    ),
                    done= b_dones[mb_indices],
                    action= b_actions.long()[mb_indices]
                )

                

                logratio = new_log_prob - b_log_probs[mb_indices]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.ppo_config["clip_coef"], 1 + self.ppo_config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                #v_loss = 0.5 * ((newvalue - b_returns[mb_indices]) ** 2).mean()
                v_loss = 0.5 * torch.abs(newvalue - b_returns[mb_indices]).mean()


                entropy_loss = entropy.mean()

                loss = pg_loss - self.ppo_config["entropy_coef"] * entropy_loss + v_loss * self.ppo_config["value_coef"]
                
                self.logger.addMetric(metric_name= "pg_loss" , value= pg_loss.item())
                self.logger.addMetric(metric_name= "value_loss", value= v_loss.item())
                self.logger.addMetric(metric_name= "entropy_loss", value= entropy_loss.item())
                agent_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.ppo_config["max_grad_norm"])
                agent_optimizer.step()



    def runUpdate(self, agent_optimizer, prev_state):
        num_steps,num_envs = self.ppo_config["num_steps"], self.ppo_config["num_envs"]


        next_obs = prev_state["next_obs"]
        next_invalid_action_masks = prev_state["next_invalid_action_masks"]
        next_done = prev_state["next_done"]
        next_lstm_state = prev_state["next_lstm_state"]



        for step in range(num_steps):
            self.storage["obs"][step] = next_obs
            self.storage["dones"][step] = next_done

            with torch.no_grad():
                action,log_prob,_,value,next_lstm_state = self.agent.getActionAndValue(x = next_obs,invalid_action_mask= next_invalid_action_masks,lstm_state= next_lstm_state,done= next_done)

                # put episode logging code here
                self.storage["actions"][step] = action
                self.storage["log_probs"][step] = log_prob
                self.storage["values"][step] = value.flatten()

                env_step = self.envs.step(action.cpu().numpy())
                next_obs, reward, next_done, next_invalid_action_masks = env_step["next_obs"], env_step["reward"], env_step["done"], env_step["invalid_action_mask"]
                self.storage["rewards"][step] = torch.tensor(reward).to(self.device).float()

                next_obs, next_done = torch.tensor(next_obs).to(self.device).float(), torch.tensor(next_done).to(self.device)
                next_invalid_action_masks = torch.tensor(next_invalid_action_masks).to(self.device)
                
        returns, advantages = self._calcAdvantage(next_obs= next_obs, next_done= next_done, next_lstm_state= next_lstm_state)
        self.storage["advantages"] = advantages
        self.storage["returns"] = returns


        prev_state = dict(
            next_obs = next_obs,
            next_done = next_done,
            next_invalid_action_masks = next_invalid_action_masks,
            next_lstm_state = next_lstm_state
        )
        self._updateModel(agent_optimizer= agent_optimizer)

        return prev_state
    

    def train(self, num_updates = 1, lr = 3e-4):
        
        agent_optimizer = torch.optim.Adam(params= self.agent.parameters(), lr= lr)

        num_steps,num_envs = self.ppo_config["num_steps"], self.ppo_config["num_envs"]
        reset = self.envs.reset()

        next_obs = torch.tensor(reset["next_obs"]).to(self.device).float()
        next_invalid_action_masks = torch.tensor(reset["invalid_action_mask"]).to(self.device)
        next_done = torch.zeros(self.ppo_config["num_envs"]).to(self.device)

        next_lstm_state = (
        torch.zeros(self.agent.lstm.num_layers, num_envs, self.agent.lstm.hidden_size).to(self.device),
        torch.zeros(self.agent.lstm.num_layers, num_envs, self.agent.lstm.hidden_size).to(self.device),
        )

        prev_state = dict(
            next_obs = next_obs,
            next_lstm_state = next_lstm_state,
            next_done = next_done,
            next_invalid_action_masks = next_invalid_action_masks
        )


        for update in range(num_updates):
            self.storage = self._initializeStorage()
            prev_state = self.runUpdate(agent_optimizer = agent_optimizer, prev_state= prev_state)
            if update%100 == 0:
                info = self.envs.getInfo()

                self.logger.addMetric(metric_name= "avg_reward", value= info["reward"])
                self.logger.addMetric(metric_name= "avg_length", value= info["length"])
                self.logger.addMetric(metric_name= "avg_profit", value= info["profit"])
                self.logger.addMetric(metric_name= "avg_trades", value= info["trades"])

            
            