import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell

import utils
from agent.ddpg import DDPGAgent


class ContextModel(nn.Module):
    def __init__(self, obs_dim, act_dim, context_dim, hidden_dim, device='cuda'):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.model = GRUCell(obs_dim+act_dim, hidden_dim)
        self.context_head = nn.Sequential(nn.ReLU(),
                                          nn.Linear(hidden_dim, context_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, acts, hidden=None):
        if hidden is None:
            hidden = torch.zeros((obs.shape[0], self.hidden_dim), device=self.device)
        for i in range(obs.shape[1]):
            hidden = self.model(torch.cat([obs[:, i], acts[:, i]], dim=-1),
                                hidden)
        context_pred = self.context_head(hidden)
        return context_pred

class PEACAgent(DDPGAgent):
    def __init__(self, update_encoder,
                 context_dim, **kwargs):
        super().__init__(**kwargs)
        self.update_encoder = update_encoder
        self.context_dim = context_dim
        print('context dim:', self.context_dim)

        self.task_model = ContextModel(self.obs_dim, self.action_dim, self.context_dim,
                                    self.hidden_dim, device=self.device).to(self.device)

        # optimizers
        self.task_opt = torch.optim.Adam(self.task_model.parameters(), lr=self.lr)

        self.task_model.train()

    def update_task_model(self, obs, action, next_obs, embodiment_id, pre_obs, pre_acts):
        metrics = dict()
        task_pred = self.task_model(pre_obs, pre_acts)
        # print(task_pred.shape)
        # print(torch.sum(embodiment_id))
        loss = F.cross_entropy(task_pred, embodiment_id.reshape(-1))

        self.task_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.task_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['task_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, embodiment_id, pre_obs, pre_acts):
        B, _ = action.shape
        task_pred = self.task_model(pre_obs, pre_acts) # B, task_num
        # calculate the task model predict prob
        task_pred = F.log_softmax(task_pred, dim=1)
        intr_rew = task_pred[torch.arange(B), embodiment_id.reshape(-1)]  # B
        task_rew = -intr_rew.reshape(B, 1)
        return task_rew

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, embodiment_id, his_o, his_a = \
            utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs, embodiment_id,
                                                       his_o, his_a)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        metrics.update(self.update_task_model(obs.detach(), action, next_obs, embodiment_id,
                                              his_o, his_a))

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
