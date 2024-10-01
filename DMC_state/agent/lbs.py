import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import utils
from agent.ddpg import DDPGAgent


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


class LBS_PRED(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.pred_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        pred_kl = self.pred_net(torch.cat([obs, action], dim=-1))
        pred_kl = D.Independent(D.Normal(pred_kl, 1.0), 1)

        return pred_kl


class LBSAgent(DDPGAgent):
    def __init__(self, lbs_scale, update_encoder, **kwargs):
        super().__init__(**kwargs)
        self.lbs_scale = lbs_scale
        self.update_encoder = update_encoder

        self.lbs = LBS(self.obs_dim, self.action_dim,
                       self.hidden_dim).to(self.device)
        # optimizers
        self.lbs_opt = torch.optim.Adam(self.lbs.parameters(), lr=self.lr)

        self.lbs_pred = LBS_PRED(self.obs_dim, self.action_dim,
                                 self.hidden_dim).to(self.device)
        self.lbs_pred_opt = torch.optim.Adam(self.lbs_pred.parameters(), lr=self.lr)

        self.lbs.train()
        self.lbs_pred.train()

    def update_lbs(self, obs, action, next_obs, step):
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

        kl_pred = self.lbs_pred(obs, action, next_obs)
        lbs_pred_loss = -kl_pred.log_prob(kl_div.detach()).mean()
        self.lbs_pred_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        lbs_pred_loss.backward()
        self.lbs_pred_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['lbs_loss'] = lbs_loss.item()
            metrics['lbs_pred_loss'] = lbs_pred_loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        kl_pred = self.lbs_pred(obs, action, next_obs)

        reward = kl_pred.mean * self.lbs_scale
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, task_id = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_lbs(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)

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
