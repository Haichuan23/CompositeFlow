import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
# Ensure this import points to the correct location of your FlowMatching class
from .guided_flow.flow_matching import FlowMatching
import wandb
import json
import os # Import os for path checking in load

# --- Constants --- (Optional but good practice)
LOG_STD_MIN = -20
LOG_STD_MAX = 2

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    """Unsqueezes and repeats tensor along specified dim."""
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class Scalar(nn.Module):
    """Learnable scalar parameter."""
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class TanhTransform(Transform):
    """
    Transform via the mapping :math:`y = \tanh(x)`.
    Numerically stable implementation.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        # Clamp to avoid log(0) or log(negative) RuntimeErrors for values slightly outside (-1, 1)
        # due to numerical precision.
        x_clamped = torch.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7)
        return 0.5 * (x_clamped.log1p() - (-x_clamped).log1p())


    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # More numerically stable computation using softplus
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class MLPNetwork(nn.Module):
    """Simple multi-layer perceptron network."""
    def __init__(self, input_dim, output_dim, hidden_size=256, activation=nn.ReLU):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):
    """Gaussian policy network with Tanh transform for bounded actions (SAC style)."""
    def __init__(self, state_dim, action_dim, max_action,
                 hidden_size=256, activation=nn.ReLU,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Network outputs mean and log_std (concatenated)
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size, activation)

    def forward(self, state, get_logprob=False, repeat=None):
        if repeat is not None:
            state = extend_and_repeat(state, 1, repeat)

        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=-1)

        # Clamp log_std for stability
        logstd = torch.clamp(logstd, self.log_std_min, self.log_std_max)
        std = logstd.exp()

        # Create distribution and apply Tanh transform
        base_dist = Normal(mu, std)
        # The cache_size=1 is important for numerical stability with TanhTransform
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(base_dist, transforms)

        # Sample action (rsample for reparameterization trick)
        # Tanh prevents actions from going much beyond +/- 1, scale by max_action
        raw_action = dist.rsample() # Action in range (-1, 1)
        action = raw_action * self.max_action # Scale to env limits

        logprob = None
        if get_logprob:
            # Calculate log prob of the action *in the transformed space*
            # Sum across the action dimension
            logprob = dist.log_prob(raw_action).sum(axis=-1, keepdim=True)

        # Calculate the mean action (deterministic output)
        # Apply Tanh to the mean of the Normal distribution
        mean_action = torch.tanh(mu) * self.max_action

        return action, logprob, mean_action


class DoubleQFunc(nn.Module):
    """Double Q-network (TD3/SAC style)."""
    def __init__(self, state_dim, action_dim, hidden_size=256, activation=nn.ReLU):
        super().__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size, activation)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size, activation)

    def forward(self, state, action):
        # Handle potential repeated actions for ensemble methods or CQL
        multiple_actions = False
        batch_size = state.shape[0]
        if action.ndim == 3 and state.ndim == 2:
            # state: [B, state_dim], action: [B, N, action_dim]
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(-1, state.shape[-1]) # [B*N, state_dim]
            action = action.reshape(-1, action.shape[-1]) # [B*N, action_dim]
        # else: state: [B, state_dim], action: [B, action_dim]

        x = torch.cat([state, action], dim=-1) # [B*N or B, state_dim + action_dim]

        q1 = self.network1(x).squeeze(-1) # [B*N or B]
        q2 = self.network2(x).squeeze(-1) # [B*N or B]

        if multiple_actions:
            q1 = q1.reshape(batch_size, -1) # [B, N]
            q2 = q2.reshape(batch_size, -1) # [B, N]

        return q1, q2


class VFlowPolicy(object):
    """
    VFlow Policy combining SAC with Flow Matching for dynamics adaptation.
    Assumes mode 1 (offline-online) training structure from train.py.
    """
    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device

        # Extract relevant config parameters with defaults
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.max_action = config['max_action']
        self.discount = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.update_interval = config.get('update_interval', 1) # How often to run updates per env step (if online)
        self.start_gate_src_sample = config.get('start_gate_src_sample', 100000.0)
        self.dynamics_train_freq = config.get('dynamics_train_freq', 5000)
        self.upsample_src = config.get('upsample_src', False)
        self.use_weight = config.get('use_weight', False)
        self.use_sample_level = config.get('use_sample_level', False)
        self.filter_percent = config.get('filter_percent', 0.7)
        self.beta = config.get('beta', 1.0)
        self.dynamics_gap_reward_scale = config.get('dynamics_gap_reward_scale', 0.0)
        self.n_samples_gap = config.get('n_samples', 100)
        self.temperature_opt = config.get('temperature_opt', False)
        self.backup_entropy = config.get('backup_entropy', True)
        # Weight for BC loss component in policy update
        # Using 'weight' from config as specified in train.py args
        self.policy_bc_weight = config.get('weight', 2.5)

        self.total_it = 0 # Total gradient steps taken

        # Dynamics Model (FlowMatching)
        # Pass necessary sub-config or the whole config to FlowMatching
        self.dynamics_model = FlowMatching(config, device) # Ensure FlowMatching uses the config correctly

        # Critic Network (Q-functions)
        hidden_size = config.get('hidden_sizes', 256)
        self.q_funcs = DoubleQFunc(self.state_dim, self.action_dim, hidden_size=hidden_size).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs).eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config.get('critic_lr', 3e-4))

        # Actor Network (Policy)
        self.policy = Policy(self.state_dim, self.action_dim, self.max_action, hidden_size=hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get('actor_lr', 3e-4))

        # Temperature (alpha) for SAC entropy bonus
        self.target_entropy = target_entropy if target_entropy else -float(self.action_dim)
        if self.temperature_opt:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get('actor_lr', 3e-4)) # Often same LR as actor
        else:
            alpha_init = config.get('alpha', 0.2) # Use initial alpha from config if fixed
            self.log_alpha = torch.log(torch.tensor([alpha_init], device=self.device)).detach()
            self.temp_optimizer = None # No optimizer needed if alpha is fixed

        print("VFlowPolicy Initialized:")
        print(f"  Discount: {self.discount}, Tau: {self.tau}, Target Entropy: {self.target_entropy}")
        print(f"  Start Gate Step: {self.start_gate_src_sample}, Dyn Train Freq: {self.dynamics_train_freq}")
        print(f"  Use Weighting: {self.use_weight}, Filter Percent: {self.filter_percent}, Beta: {self.beta}")
        print(f"  Reward Scale: {self.dynamics_gap_reward_scale}, Use Sample Level Gap: {self.use_sample_level}")
        print(f"  Optimize Alpha: {self.temperature_opt}, Initial Alpha: {self.alpha.item():.3f}")


    @property
    def alpha(self):
        """Returns the current entropy temperature alpha."""
        return self.log_alpha.exp()

    def select_action(self, state, test=True):
        """Selects action from policy based on state."""
        with torch.no_grad():
            # Ensure state is a tensor on the correct device
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action, _, mean_action = self.policy(state_tensor)
        if test:
            # Return deterministic mean action for evaluation
            return mean_action.squeeze(0).cpu().numpy()
        else:
            # Return stochastic sample for training/exploration
            return action.squeeze(0).cpu().numpy()

    def update_target(self):
        """Performs Polyak averaging for target Q-networks."""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, dynamics_gap_batch, weight_batch, writer=None):
        """Updates the Q-functions based on the batch and target values."""
        with torch.no_grad():
            # Apply dynamics gap reward modification if applicable
            # Ensure dynamics_gap_batch has shape [batch_size] or [batch_size, 1]
            if self.total_it >= self.start_gate_src_sample and self.dynamics_gap_reward_scale != 0 and dynamics_gap_batch is not None:
                # Make sure shapes are compatible for broadcasting
                current_reward_shape = reward_batch.shape
                current_gap_shape = dynamics_gap_batch.shape
                # Try to make them broadcastable, assuming reward is [B] or [B, 1]
                if reward_batch.ndim == 1: reward_batch = reward_batch.unsqueeze(-1) # [B] -> [B, 1]
                if dynamics_gap_batch.ndim == 1: dynamics_gap_batch = dynamics_gap_batch.unsqueeze(-1) # [B] -> [B, 1]

                if reward_batch.shape == dynamics_gap_batch.shape:
                     reward_batch = reward_batch + self.dynamics_gap_reward_scale * dynamics_gap_batch
                else:
                     print(f"[Warning] Shape mismatch in reward mod: Reward {current_reward_shape}, Gap {current_gap_shape}. Skipping mod.")

            # Calculate target Q-value (SAC style)
            next_action_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, next_action_batch)
            q_target = torch.min(q_t1, q_t2) # Shape [batch_size]

             # Ensure logprobs has shape [batch_size]
            if logprobs_batch.ndim > 1: logprobs_batch = logprobs_batch.squeeze(-1)

            if self.backup_entropy:
                # Target includes entropy term: R + gamma * (min(Q_targ) - alpha * log_pi)
                value_target = reward_batch.squeeze(-1) + not_done_batch.squeeze(-1) * self.discount * (q_target - self.alpha.detach() * logprobs_batch)
            else:
                # Target without entropy term: R + gamma * min(Q_targ)
                value_target = reward_batch.squeeze(-1) + not_done_batch.squeeze(-1) * self.discount * q_target
            # value_target shape: [batch_size]

        # Calculate current Q estimates
        q_1, q_2 = self.q_funcs(state_batch, action_batch) # Shape [batch_size]

        # Calculate Q-loss (potentially weighted)
        # weight_batch should have shape [batch_size]
        loss_q1 = F.mse_loss(q_1, value_target, reduction='none') # Shape [batch_size]
        loss_q2 = F.mse_loss(q_2, value_target, reduction='none') # Shape [batch_size]

        if self.use_weight:
            if weight_batch is None or weight_batch.shape[0] != q_1.shape[0]:
                 print(f"[Warning] Weight shape mismatch or None: weight {weight_batch.shape if weight_batch is not None else 'None'}, batch {q_1.shape[0]}. Using uniform weights.")
                 final_loss = loss_q1.mean() + loss_q2.mean()
            else:
                 # Ensure weight_batch has shape [batch_size]
                 if weight_batch.ndim > 1: weight_batch = weight_batch.squeeze()
                 final_loss = (weight_batch * loss_q1).mean() + (weight_batch * loss_q2).mean()
        else:
            final_loss = loss_q1.mean() + loss_q2.mean()


        # --- Optimization Step ---
        self.q_optimizer.zero_grad()
        final_loss.backward()
        # Optional: Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(self.q_funcs.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # --- Logging ---
        if writer is not None and self.total_it % 1000 == 0: # Log less frequently maybe
            log_data = {
                'train/q1_mean': q_1.mean().item(),
                'train/q2_mean': q_2.mean().item(),
                'train/q_target_mean': q_target.mean().item(),
                'train/value_target_mean': value_target.mean().item(),
                'train/q_loss': final_loss.item(),
                'train/logprob_mean': logprobs_batch.mean().item(),
                'train/batch_reward_mean': reward_batch.mean().item(), # Log potentially modified reward
            }
            if dynamics_gap_batch is not None:
                log_data['train/dynamics_gap_mean'] = dynamics_gap_batch.mean().item()
            if self.use_weight and weight_batch is not None:
                log_data['train/weight_mean'] = weight_batch.mean().item()

            for key, value in log_data.items():
                 writer.add_scalar(key, value, self.total_it)
            if wandb:
                 wandb.log(log_data, step=self.total_it)

        return final_loss.item() # Return loss value for potential tracking


    def update_policy_and_temp(self, state_batch, src_state_batch, src_action_batch, writer=None):
        """Updates the policy (actor) and temperature (alpha) using the original adaptive weighting."""
        # Freeze Q-networks during policy update
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        # Calculate Q-values and log-probabilities for the main batch
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2) # Shape [batch_size]

        # Ensure logprobs has shape [batch_size]
        if logprobs_batch.ndim > 1: logprobs_batch = logprobs_batch.squeeze(-1)

        # --- Calculate Policy Loss Components ---

        # 1. Standard SAC loss term for the combined batch (src + tar)
        # Note: Use detached alpha for policy loss calculation as is standard in SAC
        sac_term = (self.alpha.detach() * logprobs_batch - qval_batch).mean()

        # 2. Behavior Cloning (BC) loss component (only on source states/actions)
        bc_loss = torch.tensor(0.0, device=self.device) # Default to zero
        # Only compute BC loss if there are source samples in the batch
        if src_state_batch is not None and src_state_batch.shape[0] > 0:
            # Predict actions for the source states
            pred_src_act, _, _ = self.policy(src_state_batch, get_logprob=False) # No need for logprob here
            bc_loss = F.mse_loss(pred_src_act, src_action_batch)
        else:
             # If no source samples (e.g., due to aggressive filtering), bc_loss remains 0
             pass


        # 3. Calculate the adaptive weight p_w (as in the original code)
        # Use a small epsilon for numerical stability in case mean abs Q is near zero
        qval_abs_mean = qval_batch.abs().mean().detach()
        epsilon = 1e-6
        # self.policy_bc_weight is the fixed hyperparameter 'weight' from config (e.g., 2.5)
        p_w = self.policy_bc_weight / (qval_abs_mean + epsilon)

        # 4. Combine losses: Scale the SAC term by p_w and add the BC loss
        policy_loss = p_w * sac_term + bc_loss

        # --- Policy Optimizer Step ---
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Unfreeze Q-networks
        for p in self.q_funcs.parameters():
            p.requires_grad = True

        # --- Calculate Temperature Loss (if optimizing alpha) ---
        temp_loss = None
        if self.temperature_opt and self.temp_optimizer is not None:
            # Loss: E [-alpha * (log_pi + target_entropy)]
            # Use detached logprobs for temperature update
            temp_loss_tensor = -(self.alpha * (logprobs_batch.detach() + self.target_entropy)).mean()

            # --- Temperature Optimizer Step ---
            self.temp_optimizer.zero_grad()
            temp_loss_tensor.backward()
            self.temp_optimizer.step()
            temp_loss = temp_loss_tensor.item() # Get scalar value for logging/return

        # --- Logging ---
        if writer is not None and self.total_it % 1000 == 0:
            log_data = {
                'train/policy_loss_sac_term': sac_term.item(), # Log the unscaled SAC term
                'train/policy_loss_bc': bc_loss.item(),
                'train/policy_adaptive_weight_p_w': p_w.item(),
                'train/policy_loss_total': policy_loss.item(), # Log the final combined loss
                'train/alpha': self.alpha.item(),
                'train/qval_batch_abs_mean': qval_abs_mean.item(), # Log the value used for p_w calculation
            }
            if temp_loss is not None:
                log_data['train/temp_loss'] = temp_loss

            for key, value in log_data.items():
                 writer.add_scalar(key, value, self.total_it)
            if wandb:
                wandb.log(log_data, step=self.total_it)

        # Return scalar loss values
        policy_loss_scalar = policy_loss.item()
        return policy_loss_scalar, temp_loss


    def pretrain_source_flow(self, src_replay_buffer):
        """Trains the source dynamics model using Flow Matching."""
        # Ensure FlowMatching class has this method implemented correctly
        print(f"Starting Source Flow Matching Pretraining (Iteration {self.total_it})...")
        try:
            # Pass necessary parameters from config
            self.dynamics_model.train_source_flow_matching(
                src_replay_buffer,
                holdout_ratio=self.config.get('flow_matching_holdout_ratio', 0.1),
                n_epochs=self.config.get('flow_matching_training_max_epochs_source', 50),
                batch_size=self.config.get('flow_matching_batch_size', 1024),
                lr=self.config.get('flow_matching_lr', 1e-4)
            )
            print(f"Source Flow Matching Finished.")
        except AttributeError:
             print("[Error] `train_source_flow_matching` method not found in FlowMatching class.")
        except Exception as e:
             print(f"[Error] Exception during source flow pretraining: {e}")


    def train_adaptation_flow(self, tar_replay_buffer):
        """Trains/adapts the dynamics model to the target data using Flow Matching."""
        # Ensure FlowMatching class has this method implemented correctly
        print(f"Starting Adaptation Flow Matching Training (Iteration {self.total_it})...")
        try:
            # Pass necessary parameters from config
             self.dynamics_model.train_adaptation_flow_matching(
                tar_replay_buffer, # Use target buffer for adaptation
                holdout_ratio=self.config.get('flow_matching_holdout_ratio', 0.1),
                n_epochs=self.config.get('flow_matching_training_max_epochs_adaptation', 20), # Typically fewer epochs for adaptation
                batch_size=self.config.get('flow_matching_batch_size', 1024),
                lr=self.config.get('flow_matching_lr', 1e-4), # Potentially use a smaller LR for adaptation
                eta=self.config.get('flow_matching_eta', 0.0)
            )
             print(f"Adaptation Flow Matching Finished.")
        except AttributeError:
             print("[Error] `train_adaptation_flow_matching` method not found in FlowMatching class.")
        except Exception as e:
             print(f"[Error] Exception during adaptation flow training: {e}")


    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        """Performs one training step (updates Q, policy, alpha, potentially dynamics)."""

        # --- Pretrain Dynamics Model (First Step Only) ---
        if self.total_it == 0:
            # Check if source buffer has data before pretraining
            if src_replay_buffer.size > self.config.get('flow_matching_batch_size', 1024): # Need at least one batch
                 self.pretrain_source_flow(src_replay_buffer)
            else:
                 print("[Warning] Source replay buffer too small for initial flow pretraining. Skipping.")

        self.total_it += 1
        # --- Train/Adapt Dynamics Model (Periodically) ---
        # Start adaptation training *after* the gating step and periodically
        if self.total_it >= self.start_gate_src_sample and self.total_it % self.dynamics_train_freq == 0:
             if tar_replay_buffer.size > self.config.get('flow_matching_batch_size', 1024): # Check target buffer size
                 self.train_adaptation_flow(tar_replay_buffer)
             else:
                  print(f"[Warning] Target replay buffer too small for adaptation flow training at step {self.total_it}. Skipping.")


        # --- Sample Data ---
        # Check if buffers have enough data for a batch
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            # print(f"[Info] Skipping training step {self.total_it + 1}. Not enough data in buffers (Src: {src_replay_buffer.size}, Tar: {tar_replay_buffer.size}).")
            return # Skip update if not enough data

        # Determine source sample size based on upsampling logic
        if self.upsample_src and self.total_it >= self.start_gate_src_sample:
             # Calculate how many source samples needed to likely get 'batch_size' after filtering
             # This requires an estimate of the filtering ratio (1 - filter_percent)
             # Example: If filter_percent=0.7, keep 30%. Need batch_size / 0.3 samples.
             keep_ratio = max(1.0 - self.config['filter_percent'], 0.01) # Avoid division by zero
             src_sample_size = int(batch_size / keep_ratio)
             # Also consider downsample ratio if provided in config (though its purpose here is less clear)
             src_sample_size = int(src_sample_size / self.config.get('downsample_src', 1.0))
             src_sample_size = min(src_sample_size, src_replay_buffer.size) # Don't sample more than available
             src_sample_size = max(src_sample_size, batch_size) # Sample at least batch_size
        else:
            src_sample_size = batch_size

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(src_sample_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        # --- Filtering and Weighting Logic ---
        mask = torch.ones(src_state.shape[0], dtype=torch.bool, device=self.device) # Default mask (keep all)
        weight_batch = torch.ones(src_state.shape[0] + tar_state.shape[0], device=self.device) # Default weights (all 1)
        dynamics_gap_src = None
        dynamics_gap_tar = None
        dynamics_gap_combined = None # For reward modification

        if self.total_it >= self.start_gate_src_sample:
            with torch.no_grad(): # Gap estimation should not require gradients here
                # Estimate dynamics gap for source samples
                if self.use_sample_level:
                    dynamics_gap_src = self.dynamics_model.estimate_dynamics_gap_sample_level(src_state, src_action, src_next_state)
                else: # Region level
                    if self.dynamics_gap_reward_scale != 0:
                        # Calculate combined gap if needed for reward mod
                        combined_state = torch.cat([src_state, tar_state], 0)
                        combined_action = torch.cat([src_action, tar_action], 0)
                        dynamics_gap_all = self.dynamics_model.estimate_dynamics_gap(combined_state, combined_action, n_samples=self.n_samples_gap)
                        dynamics_gap_src = dynamics_gap_all[:src_state.shape[0]]
                        dynamics_gap_tar = dynamics_gap_all[src_state.shape[0]:]
                        dynamics_gap_combined = dynamics_gap_all # Store for reward mod
                    else:
                        # Only need source gap for filtering
                        dynamics_gap_src = self.dynamics_model.estimate_dynamics_gap(src_state, src_action, n_samples=self.n_samples_gap)

                if dynamics_gap_src is not None:
                    # --- Filtering based on source gap ---
                    threshold = torch.quantile(dynamics_gap_src.to(torch.float32), self.filter_percent)
                    mask = dynamics_gap_src < threshold # Boolean mask

                    # --- Weighting based on normalized source gap (only if use_weight is True) ---
                    if self.use_weight:
                        # Normalize gap: smaller gap -> higher weight
                        gap_range = dynamics_gap_src.max() - dynamics_gap_src.min()
                        normalized_gap = (dynamics_gap_src - dynamics_gap_src.max()) / (gap_range + 1e-8) # Range <= 0
                        # Weight = exp(beta * normalized_gap)
                        src_weights = torch.exp(self.beta * normalized_gap) # Shape [src_sample_size]
                    else:
                        src_weights = torch.ones_like(dynamics_gap_src) # Uniform weights if not using weighting

                    # Apply mask to source data and weights
                    src_state = src_state[mask]
                    src_action = src_action[mask]
                    src_next_state = src_next_state[mask]
                    src_reward = src_reward[mask]
                    src_not_done = src_not_done[mask]
                    src_weights = src_weights[mask] # Filter weights
                    # Also filter the raw source gap if needed for reward mod later
                    if dynamics_gap_src is not None: dynamics_gap_src = dynamics_gap_src[mask]

                    # Combine weights (filtered source weights + target weights of 1)
                    target_weights = torch.ones(tar_state.shape[0], device=self.device)
                    weight_batch = torch.cat([src_weights, target_weights], 0) # Shape [num_masked_src + tar_batch_size]

                    # Prepare combined dynamics gap for reward modification (using filtered src gap)
                    if self.dynamics_gap_reward_scale != 0:
                         if self.use_sample_level:
                              # Re-estimate gap on the final combined batch if sample-level
                              final_state = torch.cat([src_state, tar_state], 0)
                              final_action = torch.cat([src_action, tar_action], 0)
                              final_next_state = torch.cat([src_next_state, tar_next_state], 0)
                              dynamics_gap_combined = self.dynamics_model.estimate_dynamics_gap_sample_level(final_state, final_action, final_next_state)
                         elif dynamics_gap_src is not None and dynamics_gap_tar is not None:
                              # Reconstruct from filtered source and original target gaps
                              dynamics_gap_combined = torch.cat([dynamics_gap_src, dynamics_gap_tar], 0)
                         else:
                              dynamics_gap_combined = None # Couldn't reconstruct

        # If filtering wasn't done (e.g., before start_gate or if gap estimation failed)
        # mask remains all True, src data is unchanged, weight_batch is all ones.

        # Combine final batch
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        # Ensure final weight_batch matches the combined batch size
        if weight_batch.shape[0] != state.shape[0]:
             print(f"[Warning] Final weight batch size mismatch ({weight_batch.shape[0]} vs {state.shape[0]}). Using uniform weights.")
             weight_batch = torch.ones(state.shape[0], device=self.device)


        # --- Update Networks ---
        # Update Q Functions
        self.update_q_functions(state, action, reward, next_state, not_done, dynamics_gap_combined, weight_batch, writer)

        # Update Policy and Temperature
        self.update_policy_and_temp(state, src_state, src_action, writer) # Pass filtered src data for BC loss

        # Update Target Networks
        self.update_target()

        # # Increment total gradient steps
        # self.total_it += 1


    def save(self, filename_prefix):
        """Saves the policy components and dynamics model state."""
        print(f"Saving checkpoint with prefix: {filename_prefix} at step {self.total_it}")
        try:
            # Standard RL components
            torch.save(self.q_funcs.state_dict(), filename_prefix + "_critic.pth")
            torch.save(self.q_optimizer.state_dict(), filename_prefix + "_critic_optimizer.pth")
            torch.save(self.policy.state_dict(), filename_prefix + "_actor.pth")
            torch.save(self.policy_optimizer.state_dict(), filename_prefix + "_actor_optimizer.pth")
            torch.save(self.target_q_funcs.state_dict(), filename_prefix + "_target_critic.pth")
            # Save step info (including total_it) and alpha state
            state_to_save = {'total_it': self.total_it}
            if self.temperature_opt:
                state_to_save['log_alpha'] = self.log_alpha.detach().cpu()
                if self.temp_optimizer:
                    state_to_save['temp_optimizer_state_dict'] = self.temp_optimizer.state_dict()
            torch.save(state_to_save, filename_prefix + "_step_info.pt")

            # --- Save Dynamics Model (Adaptation State) ---
            # Assumes FlowMatching class has save_adaptation method
            self.dynamics_model.save_adaptation(filename_prefix + "_dynamics_adapt.pth")

            # --- Save Config ---
            config_to_save = self.config.copy()
            config_to_save['device'] = str(self.config['device']) # Convert device to string
            with open(filename_prefix + "_config.json", 'w') as f:
                json.dump(config_to_save, f, indent=4, sort_keys=True)

        except Exception as e:
            print(f"[Error] Failed to save checkpoint {filename_prefix}: {e}")


    def load(self, filename_prefix):
        """Loads the policy components and dynamics model state."""
        print(f"Loading checkpoint with prefix: {filename_prefix}")

        # Define expected file suffixes
        suffixes = {
            "critic": "_critic.pth",
            "critic_optimizer": "_critic_optimizer.pth",
            "actor": "_actor.pth",
            "actor_optimizer": "_actor_optimizer.pth",
            "step_info": "_step_info.pt",
            "target_critic": "_target_critic.pth",
            "dynamics_adapt": "_dynamics_adapt.pth", # Load adaptation state
            "config": "_config.json"
        }
        # Check existence
        missing_files = []
        file_paths = {}
        for key, suffix in suffixes.items():
            path = filename_prefix + suffix
            file_paths[key] = path
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(f"Missing checkpoint file(s): {', '.join(missing_files)}")

        try:
            # Load standard RL components
            self.q_funcs.load_state_dict(torch.load(file_paths["critic"], map_location=self.device))
            self.q_optimizer.load_state_dict(torch.load(file_paths["critic_optimizer"])) # Optimizer state is device-agnostic
            self.policy.load_state_dict(torch.load(file_paths["actor"], map_location=self.device))
            self.policy_optimizer.load_state_dict(torch.load(file_paths["actor_optimizer"]))
            self.target_q_funcs.load_state_dict(torch.load(file_paths["target_critic"], map_location=self.device))

            # Load step info and alpha state
            step_info = torch.load(file_paths["step_info"], map_location=self.device)
            self.total_it = step_info.get('total_it', 0)
            if self.temperature_opt:
                if 'log_alpha' in step_info:
                     self.log_alpha.data = step_info['log_alpha'].to(self.device)
                if 'temp_optimizer_state_dict' in step_info and self.temp_optimizer:
                     self.temp_optimizer.load_state_dict(step_info['temp_optimizer_state_dict'])


            # --- Load Dynamics Model (Adaptation State) ---
            # Assumes FlowMatching class has load_adaptation method
            self.dynamics_model.load_adaptation(file_paths["dynamics_adapt"])

            # --- Load and Verify Config ---
            with open(file_paths["config"], 'r') as f:
                saved_config = json.load(f)

            # Basic sanity check (more thorough checks might be needed)
            mismatched_keys = []
            check_keys = ['env_name', 'state_dim', 'action_dim', 'policy'] # Example critical keys
            for key in check_keys:
                if key in saved_config and key in self.config and saved_config[key] != self.config[key]:
                     mismatched_keys.append(f"{key} (Saved: {saved_config[key]}, Current: {self.config[key]})")
            if mismatched_keys:
                 print(f"[Warning] Mismatch detected between loaded and current config: {'; '.join(mismatched_keys)}")
            # Option: Update current config with loaded one?
            # self.config.update(saved_config)

            # Ensure models are on the correct device and targets are updated
            self.q_funcs.to(self.device)
            self.policy.to(self.device)
            self.target_q_funcs.to(self.device)

            for p in self.target_q_funcs.parameters():
                p.requires_grad = False
            

            print(f"Successfully loaded checkpoint. Resuming from step {self.total_it}")

        except Exception as e:
            print(f"[Error] Failed to load checkpoint {filename_prefix}: {e}")
            raise # Re-raise exception after logging