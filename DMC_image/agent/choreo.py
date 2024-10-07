import torch.nn as nn
import torch
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
from dm_env import specs

import utils
import agent.dreamer_utils as common
from agent.dreamer import ActorCritic, stop_gradient, DreamerAgent
from agent.skill_utils import get_feat_ac, SkillActorCritic, MetaCtrlAC


# https://github.com/mazpie/choreographer
class NearestEmbedFunc(torch.autograd.Function):
    # Adapted from: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/nearest_embed.py
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """

    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
        list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


class NearestEmbedEMA(nn.Module):
    # Inspired by : https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/nearest_embed.py
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('prev_cluster', torch.zeros(n_emb))

    def forward(self, x, *args, training=False, **kwargs):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)

            n_idx_choice = emb_onehot.sum(0)
            self.prev_cluster.data.add_(n_idx_choice)
            n_idx_choice[n_idx_choice == 0] = 1

            if num_arbitrary_dims:
                flatten = x.permute(
                    1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
            else:
                flatten = x.permute(1, 0).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                n_idx_choice, alpha=1 - self.decay)
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()

            cluster_size = (self.cluster_size + self.eps) / (n + self.n_emb * self.eps) * n

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        result = x + (result - x).detach()

        return result, argmin

    def kmeans(self, x, update=False):
        metrics = dict()
        metrics['unused_codes'] = torch.sum(self.prev_cluster == 0.)
        updated = 0
        batch_size = x.shape[0]
        if update:
            with torch.no_grad():
                dims = list(range(len(x.size())))
                x_expanded = x.unsqueeze(-1)
                num_arbitrary_dims = len(dims) - 2

                for idx, eq in enumerate(self.prev_cluster):
                    if eq == 0:
                        if num_arbitrary_dims:
                            emb_expanded = self.weight.view(
                                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
                        else:
                            emb_expanded = self.weight

                        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
                        min_dist, argmin = dist.min(-1)

                        probs = min_dist / (torch.sum(min_dist, dim=0, keepdim=True) + 1e-6)
                        if probs.sum() == 0:
                            break
                        x_idx = torch.multinomial(probs, 1)
                        self.weight.data[:, idx].copy_(x[x_idx].squeeze())
                        self.embed_avg.data[:, idx].copy_(x[x_idx].squeeze())

                        updated += 1

        metrics['resampled_codes'] = updated
        self.prev_cluster.data.mul_(0.)
        return metrics


class SkillModule(nn.Module):
    def __init__(self, config, skill_dim, vq_ema=True, code_dim=16, code_resampling=True, resample_every=200):
        super().__init__()
        self.cfg = config
        self._use_amp = (config.precision == 16)
        self.device = config.device
        self._updates = 0
        self.code_resampling = code_resampling
        self.resample_every = resample_every

        self.vq_ema = vq_ema
        self.code_dim = code_dim

        self.skill_dim = skill_dim
        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch

        mlp_config = {**self.cfg.reward_head}
        mlp_config['norm'] = 'layer'
        self.comit_coef = 1e-4
        self.vq_coef = 0.05
        self.skill_encoder = common.MLP(config.rssm.deter, (code_dim), **mlp_config)
        self.skill_decoder = common.MLP(code_dim, (config.rssm.deter), **mlp_config)
        self.emb = NearestEmbedEMA(skill_dim, code_dim) if self.vq_ema else NearestEmbed(skill_dim, code_dim)
        self.all_params = list(self.emb.parameters()) + list(self.skill_encoder.parameters()) + list(
            self.skill_decoder.parameters())

        opt_config = {**self.cfg.actor_opt}

        self.latent_info = dict()
        self.skill_opt = common.Optimizer('skill', self.all_params, **opt_config, use_amp=self._use_amp)

    def update(self, seq, ):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        seq = {k: flatten(v) for k, v in seq.items()}
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                skill_loss, metrics = self.skill_loss(seq)
            metrics.update(self.skill_opt(skill_loss, self.all_params))
        self._updates += 1
        if self._updates % self.resample_every == 0:
            with torch.no_grad():
                z_e = self.skill_encoder(seq['deter']).mean
                self.latent_info.update(self.emb.kmeans(z_e, update=self.code_resampling))
        metrics.update(self.latent_info)
        return metrics

    def skill_loss(self, seq):
        z_e = self.skill_encoder(seq['deter']).mean

        if self.vq_ema:
            emb, _ = self.emb(z_e, training=True)
            recon = self.skill_decoder(emb).mean

            rec_loss = F.mse_loss(recon, seq['deter'])
            commit_loss = F.mse_loss(z_e, emb.detach())

            loss = rec_loss + self.comit_coef * commit_loss
            return loss, {'rec_loss': rec_loss, 'commit_loss': commit_loss}
        else:
            z_q, _ = self.emb(z_e, weight_sg=True)
            emb, _ = self.emb(z_e.detach())
            recon = self.skill_decoder(z_q).mean

            rec_loss = F.mse_loss(recon, seq['deter'])
            vq_loss = F.mse_loss(emb, z_e.detach())
            commit_loss = F.mse_loss(z_e, emb.detach())

            loss = rec_loss + self.vq_coef * vq_loss + self.comit_coef * commit_loss
            return loss, {'rec_loss': rec_loss, 'vq_loss': vq_loss, 'commit_loss': commit_loss}


class ChoreoAgent(DreamerAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.name = name
        # self.cfg = cfg
        # self.cfg.update(**kwargs)
        # self.obs_space = obs_space
        # self.act_spec = act_spec
        # self.tfstep = None
        # self._use_amp = (cfg.precision == 16)
        # self.device = cfg.device
        # self.act_dim = act_spec.shape[0]
        #
        # # World model
        # self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep)
        self.wm.recon_skills = True

        # Exploration
        self._env_behavior = ActorCritic(self.cfg, self.act_spec, self.tfstep)
        self.lbs = common.MLP(self.wm.inp_size, (1,), **self.cfg.reward_head).to(self.device)
        self.lbs_opt = common.Optimizer('lbs', self.lbs.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
        self.lbs.train()

        # Skills
        self.skill_dim = kwargs['skill_dim']
        self.skill_pbe = utils.PBE(utils.RMS(self.device), kwargs['knn_clip'], kwargs['knn_k'], kwargs['knn_avg'],
                                   kwargs['knn_rms'], self.device)

        self._task_behavior = SkillActorCritic(self.cfg, self.act_spec, self.tfstep, self.skill_dim, )
        self.skill_module = SkillModule(self.cfg, self.skill_dim, code_dim=kwargs['code_dim'],
                                        code_resampling=kwargs['code_resampling'],
                                        resample_every=kwargs['resample_every'])
        self.wm.skill_module = self.skill_module

        # Adaptation
        self.num_init_frames = kwargs['num_init_frames']
        self.update_task_every_step = self.update_skill_every_step = kwargs['update_skill_every_step']
        self.is_ft = False

        # Common
        self.to(self.device)
        self.requires_grad_(requires_grad=False)

    def init_meta(self):
        return self.init_meta_discrete()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta_discrete(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def finetune_mode(self):
        self.is_ft = True
        self.reward_smoothing = True
        self.cfg.actor_ent = 1e-4
        self.cfg.skill_actor_ent = 1e-4
        self._env_behavior = MetaCtrlAC(self.cfg, self.skill_dim, self.tfstep, self._task_behavior,
                                        frozen_skills=self.cfg.freeze_skills, skill_len=int(1)).to(self.device)
        self._env_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
                                                       device=self.device)
        self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
                                                         device=self.device)

    def act(self, obs, meta, step, eval_mode, state):
        # Infer current state
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

        # PT stage
        # -> LBS Exploration
        if not self.is_ft:
            if eval_mode:
                actor = self._env_behavior.actor(feat)
                action = actor.mean
            else:
                actor = self._env_behavior.actor(feat)
                action = actor.sample()
            new_state = (latent, action)
            return action.cpu().numpy()[0], new_state

        # else (IS FT)
        is_adaptation = self.is_ft and (step >= self.num_init_frames // self.cfg.action_repeat)

        # Is FT AND step > num_init_frames and reward was already found
        # -> use the meta-controller
        if is_adaptation and (not self.reward_smoothing):
            if eval_mode:
                skill = self._env_behavior.actor(feat)
                skill = skill.mode()
                action = self._task_behavior.actor(torch.cat([feat, skill], dim=-1))
                action = action.mean
            else:
                skill = self._env_behavior.actor(feat)
                skill = skill.sample()
                action = self._task_behavior.actor(torch.cat([feat, skill], dim=-1))
                action = action.sample()
            new_state = (latent, action)
            return action.cpu().numpy()[0], new_state

        # Cases:
        # 1 - is not adaptation (independently from the reward smoothing)
        # 2 - is adaptation and the reward smoothing is still active
        # -> follow randomly sampled meta['skill']
        else:
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

    def pbe_reward_fn(self, seq):
        rep = seq['deter']
        B, T, _ = rep.shape
        reward = self.skill_pbe(rep.reshape(B * T, -1), cdist=True, apply_log=False).reshape(B, T, 1)
        return reward.detach()

    def code_reward_fn(self, seq):
        T, B, _ = seq['skill'].shape
        skill_target = seq['skill'].reshape(T * B, -1)
        vq_skill = skill_target @ self.skill_module.emb.weight.T
        state_pred = self.skill_module.skill_decoder(vq_skill).mean.reshape(T, B, -1)
        reward = -torch.norm(state_pred - seq['deter'], p=2, dim=-1).reshape(T, B, 1)
        return reward

    def skill_mi_fn(self, seq):
        ce_rw = self.code_reward_fn(seq)
        ent_rw = self.pbe_reward_fn(seq)
        return ent_rw + ce_rw

    def update_lbs(self, outs):
        metrics = dict()
        B, T, _ = outs['feat'].shape
        feat, kl = outs['feat'].detach(), outs['kl'].detach()
        feat = feat.reshape(B * T, -1)
        kl = kl.reshape(B * T, -1)

        loss = -self.lbs(feat).log_prob(kl).mean()
        metrics.update(self.lbs_opt(loss, self.lbs.parameters()))
        metrics['lbs_loss'] = loss.item()
        return metrics

    def update_behavior(self, state=None, outputs=None, metrics={}, data=None):
        if outputs is not None:
            post = outputs['post']
            is_terminal = outputs['is_terminal']
        else:
            data = self.wm.preprocess(data)
            embed = self.wm.encoder(data)
            post, _ = self.wm.rssm.observe(
                embed, data['action'], data['is_first'])
            is_terminal = data['is_terminal']
        #
        start = {k: stop_gradient(v) for k, v in post.items()}
        # Train skill (module + AC)
        start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
        metrics.update(self.skill_module.update(start))
        metrics.update(self._task_behavior.update(
            self.wm, start, is_terminal, self.skill_mi_fn))
        return start, metrics

    # def update_wm(self, data, step):
    #     metrics = {}
    #     state, outputs, mets = self.wm.update(data, state=None)
    #     outputs['is_terminal'] = data['is_terminal']
    #     metrics.update(mets)
    #     return state, outputs, metrics

    def update(self, data, step):
        # Train WM
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k, v in start.items()}
        if not self.is_ft:
            # LBS exploration
            with common.RequiresGrad(self.lbs):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(self.update_lbs(outputs))
            reward_fn = lambda seq: self.lbs(seq['feat']).mean
            metrics.update(self._env_behavior.update(
                self.wm, start, data['is_terminal'], reward_fn))

            # Train skill (module + AC)
            start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
            metrics.update(self.skill_module.update(start))
            metrics.update(self._task_behavior.update(
                self.wm, start, data['is_terminal'], self.skill_mi_fn))
        else:
            self.reward_smoothing = self.reward_smoothing and (not (data['reward'] > 1e-4).any())
            self._env_behavior.reward_smoothing = self.reward_smoothing

            # Train task AC
            if not self.reward_smoothing:
                reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean
                metrics.update(self._env_behavior.update(
                    self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics

    def init_from(self, other):
        # WM
        print(f"Copying the pretrained world model")
        utils.hard_update_params(other.wm.rssm, self.wm.rssm)
        utils.hard_update_params(other.wm.encoder, self.wm.encoder)
        utils.hard_update_params(other.wm.heads['decoder'], self.wm.heads['decoder'])

        # Skill
        print(f"Copying the pretrained skill modules")
        utils.hard_update_params(other._task_behavior.actor, self._task_behavior.actor)
        utils.hard_update_params(other.skill_module, self.skill_module)
        if getattr(self.skill_module, 'emb', False):
            self.skill_module.emb.weight.data.copy_(other.skill_module.emb.weight.data)

    # def report(self, data):
    #     report = {}
    #     data = self.wm.preprocess(data)
    #     for key in self.wm.heads['decoder'].cnn_keys:
    #         name = key.replace('/', '_')
    #         report[f'openl_{name}'] = self.wm.video_pred(data, key)
    #     return report
