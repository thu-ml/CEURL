import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import GRUCell

import utils
from agent.ddpg import DDPGAgent


class TaskLBSModel(nn.Module):
    def __init__(self, obs_dim, act_dim, task_dim, hidden_dim, device='cuda'):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.model = GRUCell(obs_dim+act_dim, hidden_dim)
        self.context_head = nn.Sequential(nn.ReLU(),
                                          nn.Linear(hidden_dim, task_dim))
        self.pred_net = nn.Sequential(
            nn.Linear(task_dim + obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, pre_obs, pre_acts, obs=None, act=None, hidden=None):
        if hidden is None:
            hidden = torch.zeros((pre_obs.shape[0], self.hidden_dim), device=self.device)
        for i in range(pre_obs.shape[1]):
            hidden = self.model(torch.cat([pre_obs[:, i], pre_acts[:, i]], dim=-1),
                                hidden)
        context_pred = self.context_head(hidden)

        pred_kl = self.pred_net(torch.cat([context_pred, obs, act], dim=-1))
        pred_kl = D.Independent(D.Normal(pred_kl, 1.0), 1)
        return context_pred, pred_kl


# s_t, a_t -> z_t+1
# s_t, a_t, s_t+1 -> z_t+1
# z_t+1 -> s_t+1
class LBS(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.pri_forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.pos_forward_net = nn.Sequential(
            nn.Linear(2 * obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.reconstruction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        pri_z = self.pri_forward_net(torch.cat([obs, action], dim=-1))
        pos_z = self.pos_forward_net(torch.cat([obs, action, next_obs], dim=-1))

        reco_s = self.reconstruction_net(pos_z)

        pri_z = D.Independent(D.Normal(pri_z, 1.0), 1)
        pos_z = D.Independent(D.Normal(pos_z, 1.0), 1)
        reco_s = D.Independent(D.Normal(reco_s, 1.0), 1)

        kl_div = D.kl_divergence(pos_z, pri_z)

        reco_error = -reco_s.log_prob(next_obs).mean()
        kl_error = kl_div.mean()

        return kl_error, reco_error, kl_div.detach()


# class LBS_PRED(nn.Module):
#     def __init__(self, obs_dim, action_dim, context_dim, hidden_dim):
#         super().__init__()
#
#         self.pred_net = nn.Sequential(
#             nn.Linear(context_dim + obs_dim + action_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, 1))
#
#         self.apply(utils.weight_init)
#
#     def forward(self, context, obs, action, next_obs):
#         assert obs.shape[0] == next_obs.shape[0]
#         assert obs.shape[0] == action.shape[0]
#
#         pred_kl = self.pred_net(torch.cat([context, obs, action], dim=-1))
#         pred_kl = D.Independent(D.Normal(pred_kl, 1.0), 1)
#
#         return pred_kl


# DONE: we will sample the same trajectory at each batch
# it actually samples different trajectories
# num_workers != num_tasks
class LBS_ContextAgent(DDPGAgent):
    def __init__(self, lbs_scale, update_encoder,
                 context_dim, reward_type, **kwargs):
        super().__init__(**kwargs)
        self.update_encoder = update_encoder
        self.context_dim = context_dim
        self.lbs_scale = lbs_scale
        print('task number:', context_dim)
        self.reward_type = reward_type

        self.task_model = TaskLBSModel(self.obs_dim, self.action_dim, self.context_dim,
                                       self.hidden_dim, device=self.device).to(self.device)
        self.task_opt = torch.optim.Adam(self.task_model.parameters(), lr=self.lr)
        self.task_model.train()

        self.lbs = LBS(self.obs_dim, self.action_dim,
                       self.hidden_dim).to(self.device)
        self.lbs_opt = torch.optim.Adam(self.lbs.parameters(), lr=self.lr)

        # self.lbs_pred = LBS_PRED(self.obs_dim, self.action_dim, self.context_dim,
        #                          self.hidden_dim).to(self.device)
        # self.lbs_pred_opt = torch.optim.Adam(self.lbs_pred.parameters(), lr=self.lr)

        self.lbs.train()
        # self.lbs_pred.train()

    def update_task_model(self, obs, action, next_obs, task_id, pre_obs, pre_acts):
        metrics = dict()

        kl_error, reco_error, kl_div = self.lbs(obs, action, next_obs)

        lbs_loss = kl_error.mean() + reco_error.mean()

        self.lbs_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        lbs_loss.backward()
        self.lbs_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        task_pred, kl_pred = self.task_model(pre_obs, pre_acts, obs, action)
        # print(task_pred.shape)
        # print(torch.sum(task_id))
        task_loss = F.cross_entropy(task_pred, task_id.reshape(-1))
        lbs_pred_loss = -kl_pred.log_prob(kl_div.detach()).mean()
        loss = task_loss + lbs_pred_loss

        self.task_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.task_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['lbs_loss'] = lbs_loss.item()
            metrics['lbs_pred_loss'] = lbs_pred_loss.item()
            metrics['task_loss'] = task_loss.item()

        return metrics

    def compute_lbs_reward(self, obs, action, next_obs, task_id, pre_obs, pre_acts):
        context, kl_pred = self.task_model(pre_obs, pre_acts, obs, action)
        reward = kl_pred.mean * self.lbs_scale
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, task_id, his_o, his_a = \
            utils.to_torch(batch, self.device)
        # print(obs[:10])
        # print(his_o.shape)
        # print('lalala', task_id[:5])

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_task_model(obs, action, next_obs, task_id,
                                                  his_o, his_a))

            with torch.no_grad():
                intr_reward = self.compute_lbs_reward(obs, action, next_obs, task_id,
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
