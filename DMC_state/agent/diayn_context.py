import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell
import math
from collections import OrderedDict

from dm_env import specs
import utils
from agent.ddpg import DDPGAgent


class TaskDiaynModel(nn.Module):
    def __init__(self, obs_dim, act_dim, task_dim, hidden_dim, skill_dim, device='cuda'):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.model = GRUCell(obs_dim+act_dim, hidden_dim)
        self.context_head = nn.Sequential(nn.ReLU(),
                                          nn.Linear(hidden_dim, task_dim))
        self.skill_head = nn.Sequential(nn.Linear(obs_dim+task_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, his_obs, his_acts, hidden=None):
        if hidden is None:
            hidden = torch.zeros((his_obs.shape[0], self.hidden_dim), device=self.device)
        for i in range(his_obs.shape[1]):
            hidden = self.model(torch.cat([his_obs[:, i], his_acts[:, i]], dim=-1),
                                hidden)
        context_pred = self.context_head(hidden)
        skill_pred = self.skill_head(torch.cat([context_pred, obs], dim=-1))
        return context_pred, skill_pred


# DONE: we will sample the same trajectory at each batch
# it actually samples different trajectories
# num_workers != num_tasks
class DIAYN_ContextAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim,
                 update_encoder,
                 context_dim, reward_type, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.update_encoder = update_encoder
        self.context_dim = context_dim
        print('task number:', context_dim)
        self.reward_type = reward_type

        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        super().__init__(**kwargs)

        self.task_model = TaskDiaynModel(self.obs_dim - self.skill_dim,
                                         self.action_dim, self.context_dim,
                                         self.hidden_dim, self.skill_dim,
                                         device=self.device).to(self.device)
        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.task_opt = torch.optim.Adam(self.task_model.parameters(), lr=self.lr)

        self.task_model.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_task_model(self, obs, action, next_obs, task_id,
                          pre_obs, pre_acts, skill):
        metrics = dict()
        task_pred, skill_pred = self.task_model(obs, pre_obs, pre_acts)
        # print(task_pred.shape)
        # print(torch.sum(task_id))

        z_hat = torch.argmax(skill, dim=1)
        d_pred_log_softmax = F.log_softmax(skill_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(skill_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
            pred_z.size())[0]

        task_loss = F.cross_entropy(task_pred, task_id.reshape(-1))

        loss = task_loss + d_loss

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

    def compute_intr_reward(self, obs, action, next_obs, task_id, pre_obs, pre_acts, skill):
        z_hat = torch.argmax(skill, dim=1)
        # d_pred, task_pred = self.diayn(next_obs)
        task_pred, skill_pred = self.task_model(obs, pre_obs, pre_acts)  # B, task_num
        d_pred_log_softmax = F.log_softmax(skill_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(skill_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        skill_reward = reward.reshape(-1, 1)
        return skill_reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, task_id, his_o, his_a, skill = \
            utils.to_torch(batch, self.device)
        # print(obs[:10])
        # print(his_o.shape)
        # print('lalala', task_id[:5])

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_task_model(obs.detach(), action, next_obs, task_id,
                                                  his_o, his_a, skill))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs, task_id,
                                                       his_o, his_a, skill)

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

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

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
