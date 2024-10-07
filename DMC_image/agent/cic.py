import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common
from agent.skill_utils import *

# https://github.com/rll-research/cic/blob/master/agent/cic.py


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, self.skill_dim))

        # self.next_state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
        #                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #                                     nn.Linear(hidden_dim, self.skill_dim))

        self.pred_net = nn.Sequential(nn.Linear(2 * self.skill_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, self.skill_dim))

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, self.skill_dim))
        else:
            self.skill_net = nn.Identity()

        self.apply(utils.weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state], 1))
        return query, key


# knn_k=16, knn_avg=True, rms=True, knn_clip=0.0005
class CICAgent(DreamerAgent):
    # Contrastive Intrinsic Control (CIC)
    def __init__(self, update_skill_every_step, skill_dim,
                 project_skill, temp, lstsq_batch_size,
                 knn_rms, knn_k, knn_avg, knn_clip,
                 num_init_frames, **kwargs):
        self.reward_free = True
        self.solved_meta = None
        self.temp = temp
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.project_skill = project_skill
        self.lstsq_batch_size = lstsq_batch_size
        self.num_init_frames = num_init_frames
        kwargs["meta_dim"] = self.skill_dim
        # create actor and critic

        super().__init__(**kwargs)
        # create cic first
        # self.cic = CIC(self.obs_dim - skill_dim, skill_dim,
        #                kwargs['hidden_dim'], project_skill).to(kwargs['device'])
        in_dim = self.wm.inp_size
        self.hidden_dim = in_dim
        self.cic = CIC(in_dim, skill_dim,
                       self.hidden_dim, project_skill).to(self.device)
        self._task_behavior = SFActorCritic(self.cfg, self.act_spec, self.tfstep, self.skill_dim,
                                            discrete_skills=False).to(self.device)

        # optimizers
        # self.cic_optimizer = torch.optim.Adam(self.cic.parameters(),
        #                                       lr=self.lr)
        self.cic_optimizer = common.Optimizer('cic', self.cic.parameters(),
                                              **self.cfg.model_opt, use_amp=self._use_amp)

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        self.cic.train()

    def finetune_mode(self):
        self.is_ft = True
        self.reward_free = False
        self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00,
                                                           "scale": 1.0, "eps": 1e-8}, device=self.device)
        self.cfg.actor_ent = 1e-4
        self.cfg.sf_actor_ent = 1e-4

    def act(self, obs, meta, step, eval_mode, state):
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        meta = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}

        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(latent)

        skill = meta['skill']
        inp = torch.cat([feat, skill], dim=-1)
        if eval_mode:
            actor = self._task_behavior.actor(inp)
            action = actor.mean
        else:
            actor = self._task_behavior.actor(inp)
            action = actor.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    # def init_meta(self):
    #     if not self.reward_free:
    #         # selects mean skill of 0.5 (to select skill automatically use CEM or Grid Sweep
    #         # procedures described in the CIC paper)
    #         skill = np.ones(self.skill_dim).astype(np.float32) * 0.5
    #     else:
    #         skill = np.random.uniform(0, 1, self.skill_dim).astype(np.float32)
    #     meta = OrderedDict()
    #     meta['skill'] = skill
    #     return meta

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = torch.randn(self.skill_dim)
        skill = skill / torch.norm(skill)
        skill = skill.cpu().numpy()
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        """
        Skill version:
            compute E_s[p(z, r| s)]/E_s[p(z|s)] = p(z,r) / p(z) = p(r|z) for each z
            choose the highest :D
        """
        if self.solved_meta is not None:
            return self.solved_meta
        data = next(replay_iter)
        data = self.wm.preprocess(data)
        embed = self.wm.encoder(data)
        post, prior = self.wm.rssm.observe(
            embed, data['action'], data['is_first'])
        feat = self.wm.rssm.get_feat(post)

        reward = data['reward']
        mc_returns = reward.sum(dim=1)  # B X 1

        skill_values = []
        for index in range(self.skill_dim):
            meta = dict()
            skill = torch.zeros(list(feat.shape[:-1]) + [self.skill_dim], device=self.device)
            skill[:, :, index] = 1.0
            meta['skill'] = skill

            inp = torch.cat([feat, skill], dim=-1)
            actor = self._task_behavior.actor(inp)
            a_log_probs = actor.log_prob(data['action']).sum(dim=1)

            skill_values.append((mc_returns * a_log_probs).mean())

        skill_values = torch.stack(skill_values, dim=0)
        skill_selected = torch.argmax(skill_values)

        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[skill_selected] = 1.0
        # Copy skill
        print("skill selected: ", skill_selected)
        self.solved_meta = {'skill': skill}
        self._task_behavior.solved_meta = self.solved_meta

        return self.solved_meta

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temp
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)
        return loss, cov / temperature

    # def update_cic(self, obs, skill, next_obs, step):
    #     metrics = dict()
    #
    #     loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
    #     loss = loss.mean()
    #     self.cic_optimizer.zero_grad()
    #     loss.backward()
    #     self.cic_optimizer.step()
    #
    #     if self.use_tb or self.use_wandb:
    #         metrics['cic_loss'] = loss.item()
    #         metrics['cic_logits'] = logits.norm()
    #
    #     return metrics

    def update_cic(self, skill, next_obs, step):
        metrics = dict()
        obs = torch.zeros(next_obs.shape, device=self.device)
        obs[:, 1:, :] = next_obs[:, :-1, :]
        B, T, _ = skill.shape
        skill = skill.reshape(B * T, -1)
        obs = obs.reshape(B*T, -1)
        next_obs = next_obs.reshape(B * T, -1)

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()

        metrics.update(self.cic_optimizer(loss, self.cic.parameters()))

        metrics['aps_loss'] = loss.item()
        return metrics

        # self.cic_optimizer.zero_grad()
        # loss.backward()
        # self.cic_optimizer.step()
        #
        # if self.use_tb or self.use_wandb:
        #     metrics['cic_loss'] = loss.item()
        #     metrics['cic_logits'] = logits.norm()
        #
        # return metrics

    # def compute_intr_reward(self, obs, skill, next_obs, step):
    #
    #     with torch.no_grad():
    #         loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
    #
    #     reward = loss
    #     reward = reward.clone().detach().unsqueeze(-1)
    #
    #     return reward * self.scale

    # @torch.no_grad()
    # def compute_apt_reward(self, obs, next_obs):
    #     args = APTArgs()
    #     source = self.cic.state_net(obs)
    #     target = self.cic.state_net(next_obs)
    #     reward = compute_apt_reward(source, target, args)  # (b,)
    #     return reward.unsqueeze(-1)  # (b,1)
    @torch.no_grad()
    def compute_apt_reward(self, seq):
        next_obs = self.cic.state_net(seq['feat'])
        obs = torch.zeros(next_obs.shape, device=self.device)
        obs[:, 1:, :] = next_obs[:, :-1, :]
        B, T, _ = obs.shape
        obs = obs.reshape(B * T, -1)
        next_obs = next_obs.reshape(B*T, -1)
        reward = self.pbe(cdist=True, source=obs, target=next_obs)
        reward = reward.reshape(B, T, 1)
        return reward

    def update(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k, v in start.items()}

        if self.reward_free:
            skill = data['skill']
            feat = self.wm.rssm.get_feat(start)
            with common.RequiresGrad(self.cic):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(self.update_cic(skill, feat, step))
            reward_fn = lambda seq: self.compute_apt_reward(seq)
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
            if self.solved_meta is None:
                return state, metrics

        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics

    # def update(self, replay_iter, step):
    #     metrics = dict()
    #
    #     if step % self.update_every_steps != 0:
    #         return metrics
    #
    #     batch = next(replay_iter)
    #
    #     obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
    #         batch, self.device)
    #
    #     with torch.no_grad():
    #         obs = self.aug_and_encode(obs)
    #
    #         next_obs = self.aug_and_encode(next_obs)
    #
    #     if self.reward_free:
    #         if self.update_rep:
    #             metrics.update(self.update_cic(obs, skill, next_obs, step))
    #
    #         intr_reward = self.compute_apt_reward(next_obs, next_obs)
    #
    #         reward = intr_reward
    #     else:
    #         reward = extr_reward
    #
    #     if self.use_tb or self.use_wandb:
    #         if self.reward_free:
    #             metrics['extr_reward'] = extr_reward.mean().item()
    #             metrics['intr_reward'] = intr_reward.mean().item()
    #         metrics['batch_reward'] = reward.mean().item()
    #
    #     # extend observations with skill
    #     obs = torch.cat([obs, skill], dim=1)
    #     next_obs = torch.cat([next_obs, skill], dim=1)
    #
    #     # update critic
    #     metrics.update(
    #         self.update_critic(obs, action, reward, discount, next_obs, step))
    #
    #     # update actor
    #     metrics.update(self.update_actor(obs, step))
    #
    #     # update critic target
    #     utils.soft_update_params(self.critic, self.critic_target,
    #                              self.critic_target_tau)
    #
    #     return metrics

    @torch.no_grad()
    def estimate_value(self, start, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.wm.rssm.get_feat(start)
        start['action'] = torch.zeros_like(actions[0], device=self.device)
        seq = {k: [v] for k, v in start.items()}
        for t in range(horizon):
            action = actions[t]
            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.wm.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        reward = self.wm.heads['reward'](seq['feat']).mean
        if self.cfg.mpc_opt.use_value:
            B, T, _ = seq['feat'].shape
            seq['skill'] = torch.from_numpy(self.solved_meta['skill']).repeat(B, T, 1).to(self.device)
            value = self._task_behavior._target_critic(get_feat_ac(seq)).mean
            value = torch.einsum('tbk,tbk->tb', value, seq['skill']).unsqueeze(-1)
        else:
            value = torch.zeros_like(reward, device=self.device)
        disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)

        lambda_ret = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0)

        # First step is lost because the reward is from the start state
        return lambda_ret[1]

    @torch.no_grad()
    def plan(self, obs, meta, step, eval_mode, state, t0=True):
        """
        Plan next action using Dyna-MPC inference.
        """
        if self.solved_meta is None:
            return self.act(obs, meta, step, eval_mode, state)

        # Get Dreamer's state and features
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        post, prior = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(post)

        # Sample policy trajectories
        num_pi_trajs = int(self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples)
        if num_pi_trajs > 0:
            start = {k: v.repeat(num_pi_trajs, *list([1] * len(v.shape))) for k, v in post.items()}
            img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(num_pi_trajs, 1).to(self.device)
            seq = self.wm.imagine(self._task_behavior.actor, start, None, self.cfg.mpc_opt.horizon, task_cond=img_skill)
            pi_actions = seq['action'][1:]

        # Initialize state and parameters
        start = {k: v.repeat(self.cfg.mpc_opt.num_samples + num_pi_trajs, *list([1] * len(v.shape))) for k, v in
                 post.items()}
        mean = torch.zeros(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        std = 2 * torch.ones(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.mpc_opt.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(self.cfg.mpc_opt.horizon, self.cfg.mpc_opt.num_samples, self.act_dim,
                                              device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(start, actions, self.cfg.mpc_opt.horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.mpc_opt.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                        score.sum(0) + 1e-9))
            _std = _std.clamp_(self.cfg.mpc_opt.min_std, 2)
            mean, std = self.cfg.mpc_opt.momentum * mean + (1 - self.cfg.mpc_opt.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        new_state = (post, a.unsqueeze(0))
        return a.cpu().numpy(), new_state