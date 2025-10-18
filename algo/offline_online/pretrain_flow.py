import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
from .guided_flow.flow_matching import FlowMatching
import wandb


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, 
                log_std_multiplier = 1.0,  
                log_std_offset = -1.0,
                hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)


    def forward(self, x, get_logprob=False,  repeat=None):
        #x = x + 0.0  # <--- force -0.0 to 0.0
        #print(f"The shape of x is {x.shape}")
        if repeat is not None:
            x = extend_and_repeat(x, 1, repeat)
        #print(f"The shape of x after extend_and_repeat is {x.shape}")
        # print(f"The shape of x is {s.shape}")
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=-1)
        logstd = self.log_std_multiplier() * logstd + self.log_std_offset()

        #mu = torch.clamp(mu, min=-1e6, max=1e6)
        logstd = torch.clamp(logstd, -20, 2)
        #print('logstd: ', logstd)
        std = logstd.exp()
        #std = torch.clamp(std, min=1e-6)
        #print('std: ', std)
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return action * self.max_action, logprob, mean * self.max_action

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        multiple_actions = False
        batch_size = state.shape[0]
        if action.ndim == 3 and state.ndim == 2:
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(
                -1, state.shape[-1]
            )
            action = action.reshape(-1, action.shape[-1])
        x = torch.cat([state, action], dim=-1)
        q1 = torch.squeeze(self.network1(x), dim=-1)
        q2 = torch.squeeze(self.network2(x), dim=-1)
        if multiple_actions:
            q1 = q1.reshape(batch_size, -1)
            q2 = q2.reshape(batch_size, -1)
        return q1, q2


class PretrainFlowPolicy(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']
        self.start_gate_src_sample = config['start_gate_src_sample']

        self.total_it = 0
        self.dynamics_train_freq = config['dynamics_train_freq']
        self.upsample_src = config['upsample_src']
        # self.dynamics_train_start = config['dynamics_train_start']

        self.dynamics_model = FlowMatching(config, device)
        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka temperature
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([self.config['alpha']])).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
    
    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            if self.config['dynamics_gap_reward_scale'] != 0:
                wandb.log({
                    'train/q_reward_batch': reward_batch.mean(),
                    'train/q_dynamics_gap_region_level': self.dynamics_gap_region_level.mean(),
                }, step=self.total_it)
            ##modify reward accroding to dynamics gap
            if self.config['dynamics_gap_reward_scale'] != 0:
                reward_batch = reward_batch + self.config['dynamics_gap_reward_scale'] * self.dynamics_gap_region_level.unsqueeze(1)
            # print(f"The shape of nextstate_batch is {nextstate_batch}")
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            if self.config['backup_entropy']:
                value_target = reward_batch.squeeze() + not_done_batch.squeeze() * self.discount * (q_target.squeeze() - self.alpha * logprobs_batch.squeeze())
            else:
                value_target = reward_batch.squeeze() + not_done_batch.squeeze() * self.discount * q_target.squeeze()
            # value_target = reward_batch + not_done_batch * self.discount * q_target


        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
            # wandb.log({
            #     'train/q1': q_1.mean(),
            #     'train/logprob': logprobs_batch.mean()
            # }, step=self.total_it)
        ## src always in front of tar
        if state_batch.shape[0] == self.weight.shape[0]:
            weight_temp = self.weight
        else:
            ### [src, tar] --> [tar], take the last part of the weight (length = state_batch.shape[0])
            weight_temp = self.weight[-state_batch.shape[0]:]
            #print(f"The weight_temp is {weight_temp}")

        #print(f"loss shape: {F.mse_loss(q_1, value_target, reduction='none').shape}")
        if self.config['use_weight']:
            loss = weight_temp * F.mse_loss(q_1, value_target, reduction='none') + weight_temp * F.mse_loss(q_2, value_target, reduction='none')
        else:
            # print(f"The shape of q_1 is {q_1.shape}")
            # print(f"The shape of value_target is {value_target.shape}")
            # print(f"The shape of q_2 is {q_2.shape}")
            loss = F.mse_loss(q_1, value_target, reduction='none') + F.mse_loss(q_2, value_target, reduction='none')
        
        loss = loss.mean()
        return loss

    
    def update_cql_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            if self.config['cql_max_target_backup']:
                nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True, repeat=self.config['cql_n_actions'])
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
                q_target, max_target_indices = torch.max(torch.min(q_t1, q_t2), dim=-1)
                logprobs_batch = torch.gather(logprobs_batch, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
            else:
                nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
                # take min to mitigate positive bias in q-function training
                q_target = torch.min(q_t1, q_t2)
            if self.config['backup_entropy']:
                q_target = q_target.squeeze() - self.alpha * logprobs_batch.squeeze()
            
            #print(f"The shape of reward_batch is {reward_batch.shape}")
            if self.config['dynamics_gap_reward_scale'] != 0:
                wandb.log({
                    'train/cql_reward_batch': reward_batch.mean(),
                    'train/cql_dynamics_gap_region_level': self.dynamics_gap_region_level.mean(),
                }, step=self.total_it)
               


            if self.config['dynamics_gap_reward_scale'] != 0:
                if state_batch.shape[0] == self.weight.shape[0]:
                    reward_batch = reward_batch + self.config['dynamics_gap_reward_scale'] * self.dynamics_gap_region_level.unsqueeze(1)
                else:
                    ## only src data use cql loss
                    reward_batch = reward_batch + self.config['dynamics_gap_reward_scale'] * self.dynamics_gap_region_level[:state_batch.shape[0]].unsqueeze(1)
            #print(f"after adding dynamics gap reward, the shape of reward_batch is {reward_batch.shape}")

            value_target = reward_batch.squeeze() + not_done_batch.squeeze() * self.discount * q_target.squeeze()
        
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        #print(f"The shape of q_1 is {q_1.shape}")
        #print(f"The shape of q_2 is {q_2.shape}")
        #print(f"The shape of value_target is {value_target.shape}")
        if self.config['use_weight']:
            if state_batch.shape[0] == self.weight.shape[0]:
                weight_temp = self.weight
            else:
                ### [src, tar] --> [src], take the first part of the weight (length = state_batch.shape[0])
                weight_temp = self.weight[:state_batch.shape[0]]
            loss = weight_temp * F.mse_loss(q_1, value_target) + weight_temp * F.mse_loss(q_2, value_target)
        else:
            loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)

        # add CQL loss
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.config['cql_n_actions'], action_dim), requires_grad=False).uniform_(-1, 1)
        #print('state_batch.shape: ', state_batch.shape)
        #print('action_batch.shape: ', action_batch.shape)
        cql_current_actions, cql_current_log_prob, _ = self.policy(state_batch, get_logprob=True, repeat=self.config['cql_n_actions'])
        cql_next_actions,    cql_next_log_prob,    _ = self.policy(nextstate_batch, get_logprob=True, repeat=self.config['cql_n_actions'])
        cql_current_actions, cql_current_log_prob = (cql_current_actions.detach(), cql_current_log_prob.detach())
        cql_next_actions,    cql_next_log_prob    = (cql_next_actions.detach(), cql_next_log_prob.detach())
        
        # print(f"The shape of cql_random_actions is {cql_random_actions.shape}")
        # print(f"The shape of cql_current_actions is {cql_current_actions.shape}")
        # print(f"The shape of cql_next_actions is {cql_next_actions.shape}")
        # print(f"The shape of state_batch is {state_batch.shape}")
        cql_q1_rand,            cql_q2_rand            = self.q_funcs(state_batch, cql_random_actions)
        cql_q1_current_actions, cql_q2_current_actions = self.q_funcs(state_batch, cql_current_actions)
        cql_q1_next_actions,    cql_q2_next_actions    = self.q_funcs(state_batch, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q_1, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q_2, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.config['cql_importance_sample']:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat([cql_q1_rand - random_density,
                                    cql_q1_next_actions - cql_next_log_prob.detach().squeeze(),
                                    cql_q1_current_actions - cql_current_log_prob.detach().squeeze()], dim=1,)
            cql_cat_q2 = torch.cat([cql_q2_rand - random_density,
                                    cql_q2_next_actions - cql_next_log_prob.detach().squeeze(),
                                    cql_q2_current_actions - cql_current_log_prob.detach().squeeze()], dim=1,)

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config['cql_temp'], dim=1) * self.config['cql_temp']
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config['cql_temp'], dim=1) * self.config['cql_temp']

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(cql_qf1_ood - q_1, self.config['cql_clip_diff_min'], self.config['cql_clip_diff_max']).mean()
        cql_qf2_diff = torch.clamp(cql_qf2_ood - q_2, self.config['cql_clip_diff_min'], self.config['cql_clip_diff_max']).mean()

        if self.config['cql_lagrange']:
            alpha_prime = torch.clamp(self.alpha_prime, min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.config['cql_alpha'] * (cql_qf1_diff - self.config['cql_target_action_gap'])
            cql_min_qf2_loss = alpha_prime * self.config['cql_alpha'] * (cql_qf2_diff - self.config['cql_target_action_gap'])
            self.temp_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.temp_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.config['cql_alpha']
            cql_min_qf2_loss = cql_qf2_diff * self.config['cql_alpha']
            alpha_prime_loss = state_batch.new_tensor(0.0)
            alpha_prime = state_batch.new_tensor(0.0)

        loss += cql_min_qf1_loss.squeeze() + cql_min_qf2_loss.squeeze()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/cql q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/cql logprob', logprobs_batch.mean(), self.total_it)
        
        return loss



    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss


    def pretrain_source_flow(self, src_replay_buffer, batch_size):
        #s, a, ns, r, d = self.src_replay_buffer.sample(src_replay_buffer.size)
        # need to reset up the config
        print(f"Training Source Flow Matching")
        self.dynamics_model.train_source_flow_matching(src_replay_buffer, holdout_ratio=self.config['flow_matching_holdout_ratio'], n_epochs=self.config['flow_matching_training_max_epochs_source'], batch_size=self.config['flow_matching_batch_size'], lr=self.config['flow_matching_lr'])

        print(f"Flow Matching: Dynamic Model Finish Training")
        return
    
    def train_adaptation_flow(self, src_replay_buffer, tar_replay_buffer, batch_size):
        ## Need to change number of epochs back
        self.dynamics_model.train_adaptation_flow_matching(tar_replay_buffer, holdout_ratio=self.config['flow_matching_holdout_ratio'], n_epochs=self.config['flow_matching_training_max_epochs_adaptation'], batch_size=self.config['flow_matching_batch_size'], lr=self.config['flow_matching_lr'])
        print(f"Flow Matching: Target Dynamic Model Finish Training")
        return
    

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        
        if self.total_it == 0:
            self.pretrain_source_flow(src_replay_buffer, batch_size)
        
        self.total_it += 1

        if self.total_it < self.start_gate_src_sample:
            return
        
        if self.total_it % self.dynamics_train_freq == 0 and self.total_it >= self.start_gate_src_sample:
            self.train_adaptation_flow(src_replay_buffer, tar_replay_buffer, batch_size)

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return
        
        if self.upsample_src:
            src_sample_size = int(batch_size/(1-self.config['filter_percent'])/self.config['downsample_src'])
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(src_sample_size)
        else:
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size) # [batch_size, state_dim]

        
        if self.total_it >= self.start_gate_src_sample:
            ## output: [batch_size, 1]
            dynamics_gap_sample_level = self.dynamics_model.estimate_dynamics_gap_sample_level(src_state, src_action, src_next_state) # [batch_size]
            threshold = torch.quantile(dynamics_gap_sample_level, self.config['filter_percent']) # [1]
            mask = dynamics_gap_sample_level < threshold # [batch_size]
            normalized_gap = (dynamics_gap_sample_level - dynamics_gap_sample_level.max()) / (dynamics_gap_sample_level.max() - dynamics_gap_sample_level.min() + 1e-8)  # ∈ [-1, 0]
          
            self.weight = torch.exp(self.config['beta'] * normalized_gap)
            self.weight = self.weight.unsqueeze(1)
                        #print(f"self.weight.shape: {self.weight.shape}")


        else:
            # mask = torch.ones(src_state.shape[0])
            mask = torch.ones(src_state.shape[0], dtype=torch.bool)
            ## [batch_size, 1]
            self.weight = torch.ones(src_state.shape[0], 1, device=self.device)
            #self.weight = torch.ones_like(src_state, device=self.device)

        src_state, src_action, src_next_state = src_state[mask], src_action[mask], src_next_state[mask]
        #print(f"src_state.shape: {src_state.shape}")
        src_reward, src_not_done = src_reward[mask], src_not_done[mask]
        self.weight = self.weight[mask]
        #print(f"self.weight.shape: {self.weight.shape}")
        ## concat source weight to ones vetor of size the batch size of target
        self.weight = torch.cat([self.weight, torch.ones(tar_state.shape[0], 1, device=self.device)], 0)
        #print(f"self.weight.shape: {self.weight.shape}")
        self.weight = self.weight.squeeze()

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)
        # print(f"before dynamics gap reward, the shape of state is {state.shape}")
        # print(f"before dynamics gap reward, the shape of action is {action.shape}")
        # print(f"before dynamics gap reward, the shape of reward is {reward.shape}")
        # print(f"before dynamics gap reward, the shape of next_state is {next_state.shape}")
        # print(f"before dynamics gap reward, the shape of not_done is {not_done.shape}")
        if self.config['dynamics_gap_reward_scale'] != 0:
            self.dynamics_gap_region_level = self.dynamics_model.estimate_dynamics_gap(state, action, n_samples=self.config['n_samples']) # [src_batch_size + tar_batch_size]
            #print(f"The dynamics_gap_region_level is {self.dynamics_gap_region_level}")
        
        # print(f"after dynamics gap reward, the shape of state is {state.shape}")
        # print(f"after dynamics gap reward, the shape of action is {action.shape}")
        # print(f"after dynamics gap reward, the shape of reward is {reward.shape}")
        # print(f"after dynamics gap reward, the shape of next_state is {next_state.shape}")
        # print(f"after dynamics gap reward, the shape of not_done is {not_done.shape}")

        if self.config['tar_cql']:
            q_loss_step = self.update_cql_q_functions(state, action, reward, next_state, not_done, writer)
        else:
            tar_q_loss = self.update_q_functions(tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, writer)
            src_q_loss = self.update_cql_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done, writer)
            q_loss_step = tar_q_loss + src_q_loss

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        state = torch.cat([src_state, tar_state], 0)
        pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        if self.config['temperature_opt']:
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
