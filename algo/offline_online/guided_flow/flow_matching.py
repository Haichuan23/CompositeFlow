# Assuming flow.conditional_flow_matching and backbone.mlp are correctly importable
from .flow.conditional_flow_matching_old import get_cfm
from .backbone.mlp import ResidualMLPGuidance
from torch.utils.data import DataLoader
import tqdm
import torchdiffeq
import numpy as np
import itertools
from torch.utils.data import TensorDataset
import time
import copy # Import copy for deep copying state dicts
import torch # Add the missing torch import
import os


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {} # Keep backup for temporary switching during validation

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check if shadow[name] exists, might not if model structure changed
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
                else: # Parameter added after EMA initialization? Re-initialize shadow
                    self.shadow[name] = param.data.clone()


    def apply_shadow(self):
        # Save current parameters and replace them with EMA weights
        self.backup = {} # Clear previous backup
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        # Restore original parameters
        for name, param in self.model.named_parameters():
             if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {} # Clear backup after restoring

    def get_shadow_state_dict(self):
        # Return a state_dict based on the shadow parameters
        # This is useful if we want to save the best EMA weights
        shadow_state_dict = self.model.state_dict().copy()
        for name, param in self.model.named_parameters():
             if param.requires_grad and name in self.shadow:
                 shadow_state_dict[name] = self.shadow[name].clone()
        return shadow_state_dict


class FlowMatching:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        ## chekcpoint path = source_checkpoint_path + config['env']

        filename = f"{self.config['task_name']}_source_model_state.pth"
        source_checkpoint_path = os.path.join(".", filename)


        self.CFM_source = get_cfm('cfm', config['flow_matching_sigma'])
        self.CFM_adaptation = get_cfm('ot_cfm', config['flow_matching_adaptation_sigma'])
        # self.CFM_adaptation = get_cfm('cfm', config['flow_matching_adaptation_sigma'])
        self.source_model = ResidualMLPGuidance(d_in=config['state_dim'], cond_dim=config['action_dim']+config['state_dim'], mlp_width=config['flow_matching_hidden_dim'], num_layers=config['flow_matching_source_num_layers'], activation=config['flow_matching_activation'])
        self.adaptation_model = ResidualMLPGuidance(d_in=config['state_dim'], cond_dim=config['action_dim']+config['state_dim'], mlp_width=config['flow_matching_hidden_dim'], num_layers=config['flow_matching_adaptation_num_layers'], activation=config['flow_matching_activation'])
        self.source_model.to(self.device)
        self.adaptation_model.to(self.device)
        self._max_epochs_since_update = config['max_epochs_since_update']
        self.validation_start_epoch_source = config['validation_start_epoch_source']
        self.validation_start_epoch_adaptation = config['validation_start_epoch_adaptation']

        # EMA setup is conditional
        self.source_ema = None
        self.adaptation_ema = None
        if self.config['use_ema']:
            self.source_ema = EMA(self.source_model, decay=self.config['ema_decay'])
            self.adaptation_ema = EMA(self.adaptation_model, decay=self.config['ema_decay'])

        # Placeholders for best model states
        self.best_source_model_state_dict = None
        self.best_adaptation_model_state_dict = None
        self.best_source_val_loss = float('100')
        self.best_adaptation_val_loss = float('100')


        # Placeholders for source stats and loaded flag
        self.mean_source_sa, self.std_source_sa = None, None
        self.mean_source_ns, self.std_source_ns = None, None
        self._source_loaded_from_checkpoint = False
        self._source_trained_this_run = False # Track if training occurs

        # self.mean_adapt_sa = None
        # self.std_adapt_sa = None
        # self.mean_adapt_ns = None
        # self.std_adapt_ns = None

        if os.path.exists(source_checkpoint_path): # Check existence first
            print(f"Attempting to load pre-trained source model from: {source_checkpoint_path}")
            try:
                # --- Load to CPU First ---
                print(f"Loading checkpoint '{source_checkpoint_path}' to CPU first...")
                # Load explicitly to CPU
                state = torch.load(source_checkpoint_path, map_location='cpu')
                print(" -> Checkpoint loaded to CPU.")

                # --- Check and load model state dict ---
                model_state_dict = state.get('source_model_state_dict')
                if isinstance(model_state_dict, dict):
                    self.source_model.load_state_dict(model_state_dict)
                    self.source_model.to(self.device) # Move model to target device AFTER loading state
                    print(f" -> Loaded source model state dict and moved to {self.device}.")
                else:
                    # If the main model dict is missing or wrong type, loading failed critically
                    raise TypeError(f"Checkpoint corrupt: Expected source_model_state_dict to be a dict, but got {type(model_state_dict)}")

                # --- Check, load, and move normalization stats ---
                stats_loaded_successfully = True # Track if all expected stats are valid
                stats_to_load = ['mean_source_sa', 'std_source_sa', 'mean_source_ns', 'std_source_ns']
                print(f" -> Loading normalization stats...")
                for stat_name in stats_to_load:
                    stat_tensor_cpu = state.get(stat_name) # Get stat (on CPU)
                    if stat_tensor_cpu is not None:
                        if isinstance(stat_tensor_cpu, torch.Tensor):
                            # Set attribute on the instance and move tensor to the correct device
                            setattr(self, stat_name, stat_tensor_cpu.to(self.device))
                            # print(f"    -> Loaded and moved '{stat_name}'") # Optional detailed print
                        else:
                            # Stat exists but isn't a tensor - problematic
                            print(f" -> Warning: Loaded '{stat_name}' is not a Tensor (type: {type(stat_tensor_cpu)}). Setting to None.")
                            setattr(self, stat_name, None) # Ensure it's None if not loaded correctly
                            stats_loaded_successfully = False
                    else:
                        # Stat key wasn't found in the checkpoint file
                        print(f" -> Info: Stat '{stat_name}' not found in checkpoint. Setting to None.")
                        setattr(self, stat_name, None) # Explicitly set to None
                        stats_loaded_successfully = False # Mark as not fully loaded if any expected stat is missing

                # Print summary message for stats
                if stats_loaded_successfully:
                    print(f" -> Successfully loaded source normalization stats and moved to {self.device}.")
                else:
                    print(f" -> Warning: Some or all required source normalization stats were missing or invalid in checkpoint.")

                # --- Load EMA State Logic (Conditional) ---
                # Check if EMA is enabled in the config for this run
                if self.config.get('use_ema', False):
                    print(f" -> Checking for EMA state (use_ema is True)...")
                    # Check if EMA state actually exists in the loaded file
                    if 'source_ema_shadow' in state:
                        # Ensure the self.source_ema object exists (it should have been created earlier in __init__)
                        if self.source_ema is None:
                            print(" -> Warning: use_ema is True but self.source_ema object is None. Re-initializing EMA.")
                            self.source_ema = EMA(self.source_model, decay=self.config.get('ema_decay', 0.999))

                        # Get the shadow state dictionary from the loaded state (currently on CPU)
                        source_ema_shadow_cpu = state.get('source_ema_shadow')

                        # Verify it's a dictionary before trying to iterate
                        if isinstance(source_ema_shadow_cpu, dict):
                            print(f"    -> Found source_ema_shadow state (type: {type(source_ema_shadow_cpu)}). Restoring...")
                            try:
                                # Create the new shadow dictionary, moving each tensor to the target device
                                self.source_ema.shadow = {
                                    name: param_cpu.to(self.device)
                                    for name, param_cpu in source_ema_shadow_cpu.items()
                                }
                                print(f"    -> Restored Source EMA state and moved parameters to {self.device}.")
                            except Exception as inner_e:
                                # Catch potential errors during the tensor moving process (e.g., incompatible tensor types)
                                print(f"    -> Error processing or moving EMA shadow parameters to device: {inner_e}. Skipping EMA restore.")
                                # Optionally clear the shadow dict if restore failed partially
                                # self.source_ema.shadow = {}
                        else:
                            # If the loaded data isn't a dictionary, print a warning.
                            print(f"    -> Warning: Loaded 'source_ema_shadow' is not a dictionary (type: {type(source_ema_shadow_cpu)}). Cannot load EMA state.")
                    else:
                        # Handle case where EMA is enabled in config but not found in the checkpoint
                        print("    -> Info: 'source_ema_shadow' key not found in checkpoint file.")
                else:
                    print(f" -> Skipping EMA state load (use_ema is False).")
                # --- End EMA Loading Logic ---

                # If we reached here without critical errors (like wrong model_state_dict type)
                self._source_loaded_from_checkpoint = True # Set flag indicating successful load
                self.source_model.eval() # Set model to evaluation mode after loading
                print(" -> Source model loading process finished successfully.")

            # --- Main Exception Handling for the loading block ---
            except Exception as e:
                print(f"Error during loading process for checkpoint {source_checkpoint_path}: {e}. Model state might be incomplete. Proceeding to train source model.")
                import traceback
                traceback.print_exc() # Print full traceback for detailed debugging
                self._source_loaded_from_checkpoint = False # Ensure flag is False if any error occurred

        # --- Handle case where file doesn't exist ---
        else:
            if source_checkpoint_path: # Check if path was actually constructed
                print(f"Source checkpoint not found at the path: {source_checkpoint_path}. Source model will be trained.")
            else:
                # This case might happen if config['env_name'] was missing or invalid
                print("Source checkpoint path could not be determined (e.g., missing env_name in config). Source model will be trained.")
            # _source_loaded_from_checkpoint remains False (its default)


    def save_source(self, filepath: str):
        """Saves the current source model state and normalization stats."""
        # Check if source model was trained or loaded and stats exist
        if not self._source_loaded_from_checkpoint and not self._source_trained_this_run:
             print("Warning: Source model was neither loaded nor trained. Cannot save meaningful state.")
             return False

        # Verify required stats are present
        required_stats = ['mean_source_sa', 'std_source_sa', 'mean_source_ns', 'std_source_ns']
        if any(getattr(self, stat, None) is None for stat in required_stats):
            print(f"Error: Cannot save source state because required normalization stats are missing.")
            return False # Indicate failure

        print(f"Saving source model state to {filepath}...")
        state = {
            'config': self.config, # Include config for reference
            'source_model_state_dict': self.source_model.state_dict(),
            'mean_source_sa': self.mean_source_sa.cpu(), # Save stats on CPU
            'std_source_sa': self.std_source_sa.cpu(),
            'mean_source_ns': self.mean_source_ns.cpu(),
            'std_source_ns': self.std_source_ns.cpu(),
            # Include best validation loss if you want to store it
            'best_source_val_loss': self.best_source_val_loss,
        }

        # Save EMA state if applicable
        if self.config.get('use_ema') and self.source_ema:
             cpu_source_shadow = {name: p.cpu() for name, p in self.source_ema.shadow.items()}
             state['source_ema_shadow'] = cpu_source_shadow
             print(" -> Including source EMA state.")

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(state, filepath)
            print(f" -> Source model state successfully saved.")
            return True # Indicate success
        except Exception as e:
            print(f"Error saving source state to {filepath}: {e}")
            return False # Indicate failure


    def save_adaptation(self, filepath: str):
        state = {}
        # state = {
        #     'adaptation_model_state_dict': self.adaptation_model.state_dict(),
        #     'mean_adapt_sa': self.mean_adapt_sa.cpu(),
        #     'std_adapt_sa': self.std_adapt_sa.cpu(),
        #     'mean_adapt_ns': self.mean_adapt_ns.cpu(),
        #     'std_adapt_ns': self.std_adapt_ns.cpu(),
        # }
        state = {
            'adaptation_model_state_dict': self.adaptation_model.state_dict()
        }

        if hasattr(self, 'mean_adapt_sa'):
            state['mean_adapt_sa'] = self.mean_adapt_sa.cpu()
        if hasattr(self, 'std_adapt_sa'):
            state['std_adapt_sa'] = self.std_adapt_sa.cpu()
        if hasattr(self, 'mean_adapt_ns'):
            state['mean_adapt_ns'] = self.mean_adapt_ns.cpu()
        if hasattr(self, 'std_adapt_ns'):
            state['std_adapt_ns'] = self.std_adapt_ns.cpu()
        if self.config.get('use_ema') and self.adaptation_ema:
            state['adaptation_ema_shadow'] = {k: v.cpu() for k, v in self.adaptation_ema.shadow.items()}
        torch.save(state, filepath)


    def load_adaptation(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"[Warning] Adaptation flow checkpoint not found at {filepath}")
            return
        state = torch.load(filepath, map_location=self.device)
        self.adaptation_model.load_state_dict(state['adaptation_model_state_dict'])

        # Only load these if they exist in the checkpoint
        if 'mean_adapt_sa' in state:
            self.mean_adapt_sa = state['mean_adapt_sa'].to(self.device)
            self.std_adapt_sa = state['std_adapt_sa'].to(self.device)
            self.mean_adapt_ns = state['mean_adapt_ns'].to(self.device)
            self.std_adapt_ns = state['std_adapt_ns'].to(self.device)
        else:
            print(f"[Warning] Adaptation statistics (mean/std) not found in checkpoint: {filepath}")
        # self.mean_adapt_sa = state['mean_adapt_sa'].to(self.device)
        # self.std_adapt_sa = state['std_adapt_sa'].to(self.device)
        # self.mean_adapt_ns = state['mean_adapt_ns'].to(self.device)
        # self.std_adapt_ns = state['std_adapt_ns'].to(self.device)
        if self.config.get('use_ema') and self.adaptation_ema and 'adaptation_ema_shadow' in state:
            self.adaptation_ema.shadow = {k: v.to(self.device) for k, v in state['adaptation_ema_shadow'].items()}
        print(f"Loaded adaptation flow from {filepath}")



    # === Normalize helpers ===
    def normalize(self, a, mean, std, b=None):
        # Ensure mean and std are on the correct device
        mean = mean.to(a.device)
        std = std.to(a.device)

        if b is not None:
            concat_ab = torch.cat([a, b], dim=-1)
        else:
            concat_ab = a
        normalized = (concat_ab - mean) / std

        if b is not None:
            return normalized[:, :a.shape[1]], normalized[:, a.shape[1]:]
        else:
            return normalized

    def inverse(self, a, mean, std, b=None):
         # Ensure mean and std are on the correct device
        mean = mean.to(a.device)
        std = std.to(a.device)

        if b is not None:
            concat_ab = torch.cat([a, b], dim=-1)
        else:
            concat_ab = a

        original = concat_ab * std + mean

        if b is not None:
            return original[:, :a.shape[1]], original[:, a.shape[1]:]
        else:
            return original

    # Removed _save_best method, logic moved into training loop

    def train_source_flow_matching(self, src_replay_buffer, holdout_ratio=0.1, n_epochs=300, batch_size=512, lr=1e-4):
        

        # --- Construct Save Path ---
        # This path is primarily used if training happens and saving is needed at the end
        # Use .get for robustness in case keys are missing in config
        task_or_env_name = self.config['task_name']
        filename = f"{task_or_env_name}_source_model_state.pth"
        source_checkpoint_save_path = os.path.join(".", filename) # Save in current dir

        # --- Check if model was loaded from checkpoint ---
        if self._source_loaded_from_checkpoint:
            print("-" * 30)
            print("Pre-trained source model was loaded successfully.")

            # --- Run Validation Check on Loaded Model ---
            # <<< --- START: Added Validation Logic Here --- >>>
            print("Preparing validation data for check...")
            val_loader = None # Initialize
            stats_available = all(getattr(self, stat, None) is not None
                                  for stat in ['mean_source_sa', 'std_source_sa', 'mean_source_ns', 'std_source_ns'])

            if not stats_available:
                 print("Warning: Normalization stats missing for loaded model. Cannot run validation check.")
            else:
                # --- Prepare Validation Data (Duplicated Logic Section) ---
                # We need to sample and split data here just for this check
                try:
                    if src_replay_buffer.size == 0: raise ValueError("Source replay buffer is empty.")
                    s_full, a_full, ns_full, _, _ = src_replay_buffer.sample(src_replay_buffer.size)
                    min_data_for_split = 10 # Define a minimum reasonable size
                    if s_full.shape[0] < min_data_for_split:
                        print(f"Warning: Insufficient data ({s_full.shape[0]}) for validation split. Skipping validation check.")
                    else:
                        num_holdout = min(max(1, int(s_full.shape[0] * holdout_ratio)), max(0, s_full.shape[0] - 1))
                        if num_holdout > 0:
                            s_val, a_val, ns_val = s_full[:num_holdout], a_full[:num_holdout], ns_full[:num_holdout]
                            print(f"Using {s_val.shape[0]} samples for validation check.")
                            val_batch_size = min(batch_size * 2, s_val.shape[0])
                            val_loader = DataLoader(TensorDataset(s_val, a_val, ns_val), batch_size=val_batch_size, shuffle=False)
                        else:
                             print("Not enough data to create a validation set for checking.")
                except Exception as e:
                     print(f"Error preparing validation data: {e}. Skipping validation check.")
                     val_loader = None # Ensure loader is None if prep failed
                # --- End Validation Data Prep ---


            # --- Run Validation Logic (if loader created and stats available) ---
            if val_loader is not None and stats_available: # Double check both conditions
                print("Running validation check on loaded model...")
                self.source_model.eval() # Ensure eval mode
                val_losses = []
                ema_applied_for_val = False # Track if EMA is applied for restore

                try: # Use try/finally for reliable EMA restore
                    # Apply EMA shadow weights if EMA is enabled and was loaded
                    if self.config.get('use_ema') and self.source_ema:
                        print(" -> Applying EMA shadow weights for validation check.")
                        self.source_ema.apply_shadow()
                        ema_applied_for_val = True

                    # --- Validation Loop (Copied Logic Section) ---
                    with torch.no_grad():
                        val_iterator = tqdm.tqdm(val_loader, desc="Validating Loaded Model", leave=False, dynamic_ncols=True)
                        for s_b, a_b, ns_b in val_iterator:
                            s_b, a_b, ns_b = s_b.to(self.device, non_blocking=True), a_b.to(self.device, non_blocking=True), ns_b.to(self.device, non_blocking=True)
                            try:
                                # Normalize using loaded source stats
                                s_b_norm, a_b_norm = self.normalize(s_b, self.mean_source_sa, self.std_source_sa, a_b)
                                # Generate noise
                                ns_0 = torch.randn_like(ns_b).float()

                                # Define ODE RHS function
                                def _ode_rhs_val(t, x):
                                    t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=torch.float32)
                                    cond = torch.cat([s_b_norm, a_b_norm], dim=-1).float()
                                    return self.source_model(x.float(), t_tensor, cond)

                                # Solve ODE
                                traj = torchdiffeq.odeint(
                                    _ode_rhs_val, ns_0,
                                    torch.linspace(0, 1, self.config.get('source_flow_steps', 2), device=self.device),
                                    method='euler'
                                )
                                ns_pred_norm = traj[-1]

                                # Inverse transform using loaded source stats
                                ns_pred = self.inverse(ns_pred_norm, self.mean_source_ns, self.std_source_ns)

                                # Calculate validation loss
                                val_loss = torch.mean((ns_pred - ns_b) ** 2)
                                val_losses.append(val_loss.item())

                            except ValueError as e: # Catch normalization errors
                                 print(f"\nWarning: Error during validation check batch (likely stats missing): {e}. Skipping batch.")
                                 continue
                            except Exception as e: # Catch other errors like ODE
                                 print(f"\nWarning: Error during validation check batch: {e}. Skipping batch.")
                                 continue
                    # --- End Validation Loop ---

                    # Calculate average loss after the loop
                    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
                    print(f"Validation Check Result (Loaded Model) - Avg Val Loss: {avg_val_loss:.6f}")

                finally:
                    # Restore original weights if EMA was used
                    if ema_applied_for_val:
                        print(" -> Restoring original model weights after validation check.")
                        self.source_ema.restore()
                # --- End Validation Run ---
            else:
                 # Message if validation couldn't run
                 if not stats_available: pass # Already printed warning above
                 elif val_loader is None: pass # Already printed warning above
                 else: print("Cannot run validation check.") # General case
            # <<< --- END: Added Validation Logic Here --- >>>

            print("Skipping source model training as pre-trained model was loaded.")
            print("-" * 30)
            self.source_model.eval() # Re-ensure eval mode before returning
            return # <<< --- Exit the function here --- >>>
        # --- End 'if self._source_loaded_from_checkpoint:' block ---
        
        
        
        print("Starting source model training...")
        s, a, ns, reward, not_done = src_replay_buffer.sample_all()
        print(f"Source replay buffer size: {src_replay_buffer.size}")

        num_holdout = int(s.shape[0] * holdout_ratio)
        s_train, a_train, ns_train = s[num_holdout:], a[num_holdout:], ns[num_holdout:]
        s_val, a_val, ns_val = s[:num_holdout], a[:num_holdout], ns[:num_holdout]

        if self.mean_source_sa is None or self.std_source_sa is None:
            print("Computing source (s,a) normalizers from training data...")
            sa_train_tensor = torch.cat([s_train, a_train], dim=-1).to(self.device) # Compute on device
            self.mean_source_sa = sa_train_tensor.mean(dim=0).clone().detach()
            self.std_source_sa = sa_train_tensor.std(dim=0).clone().detach() + 1e-6
            del sa_train_tensor # Free memory

        if self.mean_source_ns is None or self.std_source_ns is None:
            print("Computing source (ns) normalizers from training data...")
            ns_train_tensor = ns_train.to(self.device) # Compute on device
            self.mean_source_ns = ns_train_tensor.mean(dim=0).clone().detach()
            self.std_source_ns = ns_train_tensor.std(dim=0).clone().detach() + 1e-6
            del ns_train_tensor # Free memory

        train_loader = DataLoader(TensorDataset(s_train, a_train, ns_train), batch_size=batch_size, shuffle=True, drop_last=True) # Added pin_memory and workers
        val_loader = DataLoader(TensorDataset(s_val, a_val, ns_val), batch_size=batch_size, shuffle=False) # Larger batch for validation

        optimizer = torch.optim.Adam(self.source_model.parameters(), lr=lr)
        self.best_source_val_loss = float(100) # Reset best loss for this training run
        self.best_source_model_state_dict = None # Reset best state
        self._epochs_since_update = 0

        training_steps = 0
        for epoch in range(n_epochs):
            start_time = time.time()
            self.source_model.train() # Set model to training mode
            train_losses = []
            for s_b, a_b, ns_b in train_loader:
                s_b, a_b, ns_b = s_b.to(self.device), a_b.to(self.device), ns_b.to(self.device)

                # Normalize data
                s_b_norm, a_b_norm = self.normalize(s_b, self.mean_source_sa, self.std_source_sa, a_b)
                ns_b_norm = self.normalize(ns_b, self.mean_source_ns, self.std_source_ns)

                ns_0 = torch.randn_like(ns_b_norm)
                t = torch.rand(ns_0.shape[0], device=self.device)
                t, ns_t, ut = self.CFM_source.sample_location_and_conditional_flow(ns_0, ns_b_norm, t)

                vt = self.source_model(ns_t, t[:, None], cond=torch.cat([s_b_norm, a_b_norm], dim=-1))
                loss = torch.mean((vt - ut) ** 2)
                train_losses.append(loss.item())

                optimizer.zero_grad() # Use set_to_none=True for potential speedup
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.source_model.parameters(), max_norm=1.0)
                optimizer.step()

                training_steps += 1
                if self.config['use_ema'] and self.source_ema and training_steps % self.config['ema_update_steps'] == 0:
                    self.source_ema.update()

            avg_train_loss = np.mean(train_losses)

            # === Validation ===
            if epoch >= self.validation_start_epoch_source:
                self.source_model.eval() # Set model to evaluation mode
                val_losses = []
                if self.config['use_ema'] and self.source_ema:
                    self.source_ema.apply_shadow() # Use EMA weights for validation

                with torch.no_grad():
                    for s_b, a_b, ns_b in val_loader:
                        s_b, a_b, ns_b = s_b.to(self.device, non_blocking=True), a_b.to(self.device, non_blocking=True), ns_b.to(self.device, non_blocking=True)
                        s_b_norm, a_b_norm = self.normalize(s_b, self.mean_source_sa, self.std_source_sa, a_b)
                        # Target next state (ground truth) stays unnormalized for final MSE calc
                        # ns_b_norm = self.normalize(ns_b, self.mean_source_ns, self.std_source_ns)

                        ns_0 = torch.randn_like(ns_b) # Noise shape matches state dim

                        def _ode_rhs(t, x):
                            t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=torch.float32)
                            cond = torch.cat([s_b_norm, a_b_norm], dim=-1)
                            # Ensure inputs to model are float32
                            return self.source_model(x.float(), t_tensor, cond.float())

                        # Solve ODE
                        traj = torchdiffeq.odeint(
                            _ode_rhs,
                            ns_0.float(), # Start from noise (normalized scale)
                            torch.linspace(0, 1, self.config['source_flow_steps'], device=self.device),
                            method='euler',
                        )

                        ns_pred_norm = traj[-1] # Prediction is in normalized space
                        ns_pred = self.inverse(ns_pred_norm, self.mean_source_ns, self.std_source_ns) # Inverse transform to original space

                        # Calculate validation loss in original data space
                        val_loss = torch.mean((ns_pred - ns_b) ** 2)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

                # Check for improvement and save best model state
                improvement = (self.best_source_val_loss - avg_val_loss) / self.best_source_val_loss if self.best_source_val_loss != 0 else float('inf')
                print(f"Improvement: {improvement:.4f}")
                if avg_val_loss < self.best_source_val_loss and improvement > 0.01: # Using 1% improvement threshold
                    self.best_source_val_loss = avg_val_loss
                    # If using EMA, save the shadow weights, otherwise save current model weights
                    if self.config['use_ema'] and self.source_ema:
                         # We are already using shadow weights due to apply_shadow()
                        self.best_source_model_state_dict = copy.deepcopy(self.source_model.state_dict())
                        # Alternatively, get shadow dict directly (safer if structure might change)
                        # self.best_source_model_state_dict = copy.deepcopy(self.source_ema.get_shadow_state_dict())
                    else:
                        self.best_source_model_state_dict = copy.deepcopy(self.source_model.state_dict())
                    self._epochs_since_update = 0
                    print(f"*** New best source model saved at epoch {epoch} with Val Loss: {avg_val_loss:.4f} ***")
                else:
                    self._epochs_since_update += 1

                # Restore original weights if EMA was used for validation
                if self.config['use_ema'] and self.source_ema:
                    self.source_ema.restore()

                # Early stopping check
                if self._epochs_since_update > self._max_epochs_since_update:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break # Stop training
            else:
                 epoch_time = time.time() - start_time
                 print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, (Validation starts at epoch {self.validation_start_epoch_source}), Time: {epoch_time:.2f}s")


        # === After Training Loop ===
        print("Source model training finished.")
        # Load the best model state found during training
        if self.best_source_model_state_dict:
            print(f"Loading best source model weights from epoch with loss {self.best_source_val_loss:.4f}")
            self.source_model.load_state_dict(self.best_source_model_state_dict)
            # If EMA was used, we might want to re-initialize EMA with the best weights
            if self.config['use_ema'] and self.source_ema:
                 print("Re-initializing EMA for source model based on best weights.")
                 self.source_ema = EMA(self.source_model, decay=self.config['ema_decay'])
            self._source_trained_this_run = True # Mark as trained (even if not 'best')

        else:
            print("Warning: No best source model state was saved (validation loss might not have improved enough or validation didn't run). Using last state.")
            self._source_trained_this_run = True # Mark as trained (even if not 'best')
       
       # --- Save the final (best) source model state ---
        if self._source_trained_this_run:
            save_success = self.save_source(source_checkpoint_save_path)
            if not save_success:
                 print("Error: Failed to save the trained source model state after training.")
        else:
            print("Source training did not complete successfully or was skipped, state not saved.")

        # Ensure model is in eval mode after training potentially
        self.source_model.eval()


    def train_adaptation_flow_matching(self, tar_replay_buffer, holdout_ratio=0.1, n_epochs=100, batch_size=512, lr=1e-4, eta=0.0):
        print("Starting adaptation model training...")
        # Reinitialize the adaptation model FRESH for each call
        print("Re-initializing adaptation model.")
        self.adaptation_model = ResidualMLPGuidance(
            d_in=self.config['state_dim'],
            cond_dim=self.config['action_dim'] + self.config['state_dim'],
            mlp_width=self.config['flow_matching_hidden_dim'],
            num_layers=self.config['flow_matching_adaptation_num_layers'],
            activation=self.config['flow_matching_activation']
        ).to(self.device)

        # Reinitialize EMA for the new adaptation model if used
        self.adaptation_ema = None
        if self.config['use_ema']:
            print("Re-initializing EMA for adaptation model.")
            self.adaptation_ema = EMA(self.adaptation_model, decay=self.config['ema_decay'])

        # Ensure source model is in eval mode and frozen
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad = False
        print("Source model frozen.")

        # Apply EMA shadow weights to the source model IF EMA is enabled and trained
        source_ema_applied = False
        if self.config['use_ema'] and self.source_ema and self.best_source_model_state_dict: # Make sure source EMA exists and was trained
            print("Applying EMA shadow weights to source model for adaptation training.")
            self.source_ema.apply_shadow()
            source_ema_applied = True


        # Unpack and split target dataset
        s, a, ns, reward, not_done = tar_replay_buffer.sample_all()
        num_holdout = int(s.shape[0] * holdout_ratio)
        s_train, a_train, ns_train = s[num_holdout:], a[num_holdout:], ns[num_holdout:]
        s_val, a_val, ns_val = s[:num_holdout], a[:num_holdout], ns[:num_holdout]

        # Compute target domain normalization constants (on training data)
        print("Computing adaptation domain normalizers...")
        sa_train = torch.cat([s_train, a_train], dim=-1)
        self.mean_adapt_sa = sa_train.mean(dim=0).clone().detach() # Use CPU tensors
        self.std_adapt_sa = sa_train.std(dim=0).clone().detach() + 1e-6

        self.mean_adapt_ns = ns_train.mean(dim=0).clone().detach() # Use CPU tensors
        self.std_adapt_ns = ns_train.std(dim=0).clone().detach() + 1e-6
        print("Adaptation normalizers computed.")


        # DataLoaders
        train_loader = DataLoader(TensorDataset(s_train, a_train, ns_train), batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(s_val, a_val, ns_val), batch_size=batch_size, shuffle=False, drop_last=False)

        # Optimizer and tracking variables
        optimizer = torch.optim.Adam(self.adaptation_model.parameters(), lr=lr)
        self.best_adaptation_val_loss = float('100') # Reset best loss
        self.best_adaptation_model_state_dict = None # Reset best state
        self._epochs_since_update = 0
        training_steps = 0

        for epoch in range(n_epochs):
            # print(f"Now I am in epoch {epoch}") # tqdm provides progress
            start_time = time.time()
            self.adaptation_model.train() # Set adaptation model to train mode
            train_losses = []

            for s_b, a_b, ns_b in train_loader:
                s_b, a_b, ns_b = s_b.to(self.device), a_b.to(self.device), ns_b.to(self.device)

                # Normalize s, a using SOURCE normalizers (condition for source model)
                s_b_norm_src, a_b_norm_src = self.normalize(s_b, self.mean_source_sa, self.std_source_sa, a_b)
                cond_src = torch.cat([s_b_norm_src, a_b_norm_src], dim=-1).float()

                # Generate initial noise
                ns_0 = torch.randn_like(ns_b).float() # Match state dim, ensure float
                
                #self.nfe = 0
                # Source flow prediction (using potentially EMA source model)
                with torch.no_grad():
                    # Source ODE RHS
                    def _source_ode_rhs(t, x):
                         #self.nfe += 1
                         t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=torch.float32)
                         return self.source_model(x.float(), t_tensor, cond_src)

                    traj_src = torchdiffeq.odeint(
                        _source_ode_rhs,
                        ns_0,
                        torch.linspace(0, 1, self.config['source_flow_steps'], device=self.device),
                        method="euler"
                    )
                #print(f"ODE solver evaluated {self.nfe} times!")
                ns_src_pred_norm_src = traj_src[-1] # Prediction in source normalized space

                # Inverse transform using SOURCE normalizers
                ns_src_pred = self.inverse(ns_src_pred_norm_src, self.mean_source_ns, self.std_source_ns)
                #print(f"ns_src_pred: {ns_src_pred}")
                #ns_src_pred = ns_src_pred.to(self.device) # Already on device

                # Normalize source prediction using ADAPTATION normalizers (start point for adaptation flow)
                ns_src_pred_norm_adapt = self.normalize(ns_src_pred, self.mean_adapt_ns, self.std_adapt_ns)
                
                s_b_norm_adapt, a_b_norm_adapt = self.normalize(s_b, self.mean_adapt_sa, self.std_adapt_sa, a_b)

                # Normalize target next state using ADAPTATION normalizers (end point for adaptation flow)
                ns_b_norm_adapt = self.normalize(ns_b, self.mean_adapt_ns, self.std_adapt_ns)

                # Adaptation training step
                t = torch.rand(ns_b_norm_adapt.shape[0], device=self.device)
                ### compute the running time needed for sampling the location and conditional flow
                sample_time = time.time()
                ### this is the time for the adaptation flow
                t, ns_t, ut = self.CFM_adaptation.sample_location_and_conditional_flow_condtion_version(ns_src_pred_norm_adapt, ns_b_norm_adapt, s_b_norm_adapt, s_b_norm_adapt, a_b_norm_adapt, a_b_norm_adapt, t)
                sample_time = time.time() - sample_time
                #print(f"Time needed for sampling the location and conditional flow: {sample_time:.2f}s")

                # Condition for adaptation model: Use SOURCE normalized s, a
                cond_adapt = cond_src # Using the same condition as source model

                # Ensure inputs are float
                ns_t, t = ns_t.float(), t.float()[:, None] # Add time dimension
                cond_adapt = cond_adapt.float()
                #print(f"ns_t: {ns_t}")
                #print(f"t: {t}")
                #print(f"cond_adapt: {cond_adapt}")
                vt = self.adaptation_model(ns_t, t, cond=cond_adapt)
                #print(f"vt: {vt}")
                #print(f"ut: {ut}")
                loss = torch.mean((vt - ut) ** 2)
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.adaptation_model.parameters(), max_norm=1.0)
                optimizer.step()

                training_steps += 1
                if self.config['use_ema'] and self.adaptation_ema and training_steps % self.config['ema_update_steps'] == 0:
                    self.adaptation_ema.update()

            avg_train_loss = np.mean(train_losses)

            # === Validation ===
            if epoch >= self.validation_start_epoch_adaptation:
                self.adaptation_model.eval() # Set adaptation model to eval mode
                val_losses = []
                if self.config['use_ema'] and self.adaptation_ema:
                    self.adaptation_ema.apply_shadow() # Use EMA weights for validation

                with torch.no_grad():
                    for s_b, a_b, ns_b in val_loader:
                        s_b, a_b, ns_b = s_b.to(self.device, non_blocking=True), a_b.to(self.device, non_blocking=True), ns_b.to(self.device, non_blocking=True)
                        ns_0 = torch.randn_like(ns_b).float()

                        # Normalize s, a using SOURCE normalizers
                        s_b_norm_src, a_b_norm_src = self.normalize(s_b, self.mean_source_sa, self.std_source_sa, a_b)
                        cond_src = torch.cat([s_b_norm_src, a_b_norm_src], dim=-1).float()

                         # Source flow prediction (using potentially EMA source model)
                        def _source_ode_rhs_val(t, x):
                            t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=torch.float32)
                            return self.source_model(x.float(), t_tensor, cond_src)

                        traj_src = torchdiffeq.odeint(
                            _source_ode_rhs_val,
                            ns_0,
                            torch.linspace(0, 1, self.config['source_flow_steps'], device=self.device),
                            method="euler"
                        )
                        ns_src_pred_norm_src = traj_src[-1]
                        ns_src_pred = self.inverse(ns_src_pred_norm_src, self.mean_source_ns, self.std_source_ns)
                        ns_src_pred_norm_adapt = self.normalize(ns_src_pred, self.mean_adapt_ns, self.std_adapt_ns)

                        # Adaptation rollout (using potentially EMA adaptation model)
                        cond_adapt = cond_src # Use same condition

                        def _adapt_ode_rhs_val(t, x):
                            t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=torch.float32)
                            return self.adaptation_model(x.float(), t_tensor, cond_adapt)

                        traj_adapt = torchdiffeq.odeint(
                            _adapt_ode_rhs_val,
                            ns_src_pred_norm_adapt.float(), # Start from source prediction (adapt-normalized)
                            torch.linspace(0, 1, self.config['adaptation_flow_steps'], device=self.device),
                            method="euler"
                        )
                        ns_pred_norm_adapt = traj_adapt[-1] # Final prediction in adapt-normalized space

                        # Inverse transform using ADAPTATION normalizers
                        ns_pred = self.inverse(ns_pred_norm_adapt, self.mean_adapt_ns, self.std_adapt_ns)

                        # Calculate validation loss in original data space
                        val_loss = torch.mean((ns_pred - ns_b) ** 2).item()
                        val_losses.append(val_loss)

                avg_val_loss = np.mean(val_losses)
                epoch_time = time.time() - start_time
                #print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

                # Check for improvement and save best model state
                improvement = (self.best_adaptation_val_loss - avg_val_loss) / self.best_adaptation_val_loss if self.best_adaptation_val_loss != 0 else float(100)

                if avg_val_loss < self.best_adaptation_val_loss and improvement > 0.01: # Using 1% improvement threshold
                    self.best_adaptation_val_loss = avg_val_loss
                    # If using EMA, save the shadow weights, otherwise save current model weights
                    if self.config['use_ema'] and self.adaptation_ema:
                        # We are already using shadow weights due to apply_shadow()
                        self.best_adaptation_model_state_dict = copy.deepcopy(self.adaptation_model.state_dict())
                        # Alternatively:
                        # self.best_adaptation_model_state_dict = copy.deepcopy(self.adaptation_ema.get_shadow_state_dict())
                    else:
                        self.best_adaptation_model_state_dict = copy.deepcopy(self.adaptation_model.state_dict())
                    self._epochs_since_update = 0
                    #print(f"*** New best adaptation model saved at epoch {epoch} with Val Loss: {avg_val_loss:.4f} ***")
                else:
                    self._epochs_since_update += 1

                 # Restore original weights if EMA was used for validation
                if self.config['use_ema'] and self.adaptation_ema:
                    self.adaptation_ema.restore()

                # Early stopping check
                if self._epochs_since_update > self._max_epochs_since_update:
                    #print(f"Early stopping triggered at epoch {epoch}.")
                    break # Stop training
            else:
                 epoch_time = time.time() - start_time
                 #print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, (Validation starts at epoch {self.validation_start_epoch_adaptation}), Time: {epoch_time:.2f}s")


        # === After Training Loop ===
        print("Adaptation model training finished.")

        # Restore source model state if EMA was applied
        if source_ema_applied:
             print("Restoring original source model weights.")
             self.source_ema.restore()

        # Unfreeze the source model parameters
        print("Unfreezing source model parameters.")
        for param in self.source_model.parameters():
            param.requires_grad = True
        self.source_model.train() # Put source model back into train mode generally

        # Load the best adaptation model state found during training
        if self.best_adaptation_model_state_dict:
            print(f"Loading best adaptation model weights from epoch with loss {self.best_adaptation_val_loss:.4f}")
            self.adaptation_model.load_state_dict(self.best_adaptation_model_state_dict)
             # If EMA was used, we might want to re-initialize EMA with the best weights
            if self.config['use_ema'] and self.adaptation_ema:
                 print("Re-initializing EMA for adaptation model based on best weights.")
                 self.adaptation_ema = EMA(self.adaptation_model, decay=self.config['ema_decay'])
        else:
            print("Warning: No best adaptation model state was saved. Using last state.")

        # Ensure adaptation model is in eval mode after training
        self.adaptation_model.eval()

        # No return needed as the models are modified in-place

# Example Usage (assuming config and device are defined, and src/tar replay buffers exist)
# flow_matcher = FlowMatching(config, device)
# flow_matcher.train_source_flow_matching(src_replay_buffer, n_epochs=config['source_epochs'])
# flow_matcher.train_adaptation_flow_matching(tar_replay_buffer, n_epochs=config['adaptation_epochs'])
# Now flow_matcher.source_model and flow_matcher.adaptation_model hold the best weights found.

    def estimate_dynamics_gap_sample_level(self, s, a, ns, model=None):
        """
        Calculates the gap between a given next state 'ns' and the result of
        running the adaptation flow *starting from* that normalized 'ns'.

        NOTE: This measures the distortion of 'ns' by the adaptation flow,
              NOT the prediction error of the full pipeline compared to 'ns'.
              Ensure this logic matches your intended use case.
        """
        # --- No changes needed to the core logic based on previous edits ---
        # --- Ensures correct model state (potentially best EMA) is used ---

        # Use adaptation_model if model is None (kept original default)
        if model is None:
            model = self.adaptation_model # Although 'model' argument is not used later? Let's assume adaptation_model is always intended.

        # Ensure models are in eval mode
        self.source_model.eval()
        self.adaptation_model.eval() # Use self.adaptation_model consistently

        s_dev = s.to(self.device)
        a_dev = a.to(self.device)
        ns_dev = ns.to(self.device)

        # Normalize s, a using source stats for conditioning
        s_norm, a_norm = self.normalize(s_dev, self.mean_source_sa, self.std_source_sa, a_dev)
        cond = torch.cat([s_norm, a_norm], dim=-1).float()

        # Normalize ns using adaptation stats as the starting point for the ODE
        ns_norm_adapt = self.normalize(ns_dev, self.mean_adapt_ns, self.std_adapt_ns).float()

        gap = None # Initialize gap
        ema_applied = False
        try: # Use try/finally to ensure EMA restore happens
            with torch.no_grad():
                if self.config['use_ema'] and self.adaptation_ema and self.source_ema: # Check if EMA exists
                    # print("Applying EMA for gap sample level estimation") # Debug print
                    self.source_ema.apply_shadow() # Source might not be used, but apply for consistency if needed later
                    self.adaptation_ema.apply_shadow()
                    ema_applied = True

                # Define adaptation ODE RHS
                def _adapt_ode_rhs(t, x):
                    t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32)
                    # Ensure model call uses self.adaptation_model
                    return self.adaptation_model(x.float(), t_tensor, cond)

                # Run adaptation flow starting from ns_norm_adapt
                traj_adaptation = torchdiffeq.odeint(
                    _adapt_ode_rhs,
                    ns_norm_adapt, # Starting point
                    torch.linspace(0, 1, self.config['adaptation_flow_steps'], device=self.device, dtype=torch.float32),
                    method="euler",
                )

                ns_adaptation_norm = traj_adaptation[-1] # Result is in adaptation-normalized space

                # Inverse transform using adaptation stats
                ns_adaptation = self.inverse(ns_adaptation_norm, self.mean_adapt_ns, self.std_adapt_ns)

                # Calculate gap between the result and the original ns
                gap = torch.norm(ns_adaptation - ns_dev, dim=-1) # Compare to ns_dev

        finally:
            # Restore original weights if EMA was applied
            if ema_applied:
                # print("Restoring EMA after gap sample level estimation") # Debug print
                self.source_ema.restore()
                self.adaptation_ema.restore()

        # Ensure model is back to train mode if needed (though usually eval after training)
        # self.source_model.train()
        # self.adaptation_model.train()

        return gap


    def estimate_dynamics_gap(self, s, a, n_samples=100, model=None):
        """
        Estimates the difference between source model prediction and adaptation
        model prediction (starting from source prediction), averaged over n_samples.
        """
         # --- Modified tensor repetition logic ---
         # --- Ensures correct model state (potentially best EMA) is used ---

        # Ensure models are in eval mode
        self.source_model.eval()
        self.adaptation_model.eval() # model argument is unused here too

        s_dev = s.to(self.device)
        a_dev = a.to(self.device)
        batch_size = s_dev.shape[0]
        state_dim = s_dev.shape[-1] # Get state dim from s

        # Normalize s, a using source stats
        s_norm, a_norm = self.normalize(s_dev, self.mean_source_sa, self.std_source_sa, a_dev)
        cond = torch.cat([s_norm, a_norm], dim=-1).float()

        # --- Improved tensor repetition ---
        # Repeat condition tensor for n_samples
        # cond shape: (batch_size, cond_dim)
        # Target cond_rep shape: (batch_size * n_samples, cond_dim)
        cond_rep = cond.repeat_interleave(n_samples, dim=0)
        # --- End of improved tensor repetition ---

        # Generate initial noise for n_samples
        # Target ns_0 shape: (batch_size * n_samples, state_dim)
        ns_0 = torch.randn(batch_size * n_samples, state_dim, device=self.device).float()

        gap = None # Initialize gap
        ema_applied = False
        try: # Use try/finally to ensure EMA restore happens
            with torch.no_grad():
                if self.config['use_ema'] and self.source_ema and self.adaptation_ema: # Check if EMA exists
                    # print("Applying EMA for dynamics gap estimation") # Debug print
                    self.source_ema.apply_shadow()
                    self.adaptation_ema.apply_shadow()
                    ema_applied = True

                # === Source Flow ===
                def _source_ode_rhs(t, x):
                    t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32)
                    # Condition uses the repeated tensor
                    return self.source_model(x.float(), t_tensor, cond_rep)

                traj_source = torchdiffeq.odeint(
                    _source_ode_rhs,
                    ns_0, # Start from noise
                    torch.linspace(0, 1, self.config['source_flow_steps'], device=self.device, dtype=torch.float32),
                    method="euler",
                )
                ns_source_norm_src = traj_source[-1] # Result in source-normalized space

                # Inverse transform using SOURCE stats
                ns_source = self.inverse(ns_source_norm_src, self.mean_source_ns, self.std_source_ns)

                # === Adaptation Flow ===
                # Normalize source prediction using ADAPTATION stats
                ns_source_norm_adapt = self.normalize(ns_source, self.mean_adapt_ns, self.std_adapt_ns)

                # Define adaptation ODE RHS
                def _adapt_ode_rhs(t, x):
                    t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32)
                     # Condition uses the repeated tensor
                    return self.adaptation_model(x.float(), t_tensor, cond_rep)

                traj_adaptation = torchdiffeq.odeint(
                    _adapt_ode_rhs,
                    ns_source_norm_adapt.float(), # Start from adapt-normalized source prediction
                    torch.linspace(0, 1, self.config['adaptation_flow_steps'], device=self.device, dtype=torch.float32),
                    method="euler",
                )
                ns_adaptation_norm_adapt = traj_adaptation[-1] # Result in adapt-normalized space

                # Inverse transform using ADAPTATION stats
                ns_adaptation = self.inverse(ns_adaptation_norm_adapt, self.mean_adapt_ns, self.std_adapt_ns)

                # Reshape predictions back to (batch_size, n_samples, state_dim)
                ns_source_reshaped = ns_source.view(batch_size, n_samples, -1)
                ns_adaptation_reshaped = ns_adaptation.view(batch_size, n_samples, -1)

                # Calculate gap: Mean norm difference over samples
                gap_per_sample = torch.norm(ns_adaptation_reshaped - ns_source_reshaped, dim=-1) # Shape (batch_size, n_samples)
                gap = torch.mean(gap_per_sample, dim=-1) # Shape (batch_size,)

        finally:
             # Restore original weights if EMA was applied
            if ema_applied:
                # print("Restoring EMA after dynamics gap estimation") # Debug print
                self.source_ema.restore()
                self.adaptation_ema.restore()

        # Ensure model is back to train mode if needed
        # self.source_model.train()
        # self.adaptation_model.train()

        return gap











    # def estimate_dynamics_gap_sample_level(self, s, a, ns, model=None):
    #     if model is None:
    #         model = self.adaptation_model

    #     s = s.to(self.device)
    #     a = a.to(self.device)
    #     ns = ns.to(self.device)


    #     s, a = self.normalize(s, self.mean_source_sa, self.std_source_sa, a)

    #     ns_norm = self.normalize(ns, self.mean_adapt_ns, self.std_adapt_ns)


    #     self.source_model.eval()
    #     self.adaptation_model.eval()

    #     with torch.no_grad():
    #         if self.config['use_ema']:
    #             self.source_ema.apply_shadow()
    #             self.adaptation_ema.apply_shadow()

    #         ## start adaptation flow matching
    #         traj_adaptation = torchdiffeq.odeint(
    #             lambda t, x: self.adaptation_model.forward(
    #                 x.float(),
    #                 torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32),
    #                 torch.cat([s, a], dim=-1).float()
    #             ),
    #             ns_norm.float(),
    #             torch.linspace(0, 1, self.config['adaptation_flow_steps'], device=self.device, dtype=torch.float32),
    #             atol=1e-4,
    #             rtol=1e-4,
    #             method="dopri5",
    #         )


    #         ns_adaptation = traj_adaptation[-1] ## [batch_size, state_dim]

    #         ns_adaptation = self.inverse(ns_adaptation, self.mean_adapt_ns, self.std_adapt_ns)

    #         gap = torch.norm(ns_adaptation - ns, dim=-1)
    #     if self.config['use_ema']:
    #         self.source_ema.restore()
    #         self.adaptation_ema.restore()
        
    #     return gap


    
    # def estimate_dynamics_gap(self, s, a, n_samples=100, model=None):
    #     if model is None:
    #         model = self.adaptation_model

    #     s = s.to(self.device)
    #     a = a.to(self.device)

    #     s, a = self.normalize(s, self.mean_source_sa, self.std_source_sa, a)


    #     ns_0 = torch.randn(s.shape[0], n_samples, s.shape[1]).to(self.device).reshape(-1,s.shape[-1])
    #     # repeat the source data n_samples times, s.shape = [batch_size, state_dim]
    #     s = s.repeat(n_samples, 1, 1)
    #     a = a.repeat(n_samples, 1, 1)

    #     s = s.permute(1,0,2)
    #     a = a.permute(1,0,2)

    #     self.source_model.eval()
    #     self.adaptation_model.eval()

    #     with torch.no_grad():
    #         if self.config['use_ema']:
    #             self.source_ema.apply_shadow()
    #             self.adaptation_ema.apply_shadow()

    #         traj = torchdiffeq.odeint(
    #             lambda t, x: self.source_model.forward(
    #                 x.float(),
    #                 torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32),
    #                 torch.cat([s, a], dim=-1).float()
    #             ),
    #             ns_0.float(),
    #             torch.linspace(0, 1, self.config['source_flow_steps'], device=self.device, dtype=torch.float32),
    #             atol=1e-4,
    #             rtol=1e-4,
    #             method="dopri5",
    #         )
            
    #         ns_source = traj[-1]

    #         ns_source = self.inverse(ns_source, self.mean_source_ns, self.std_source_ns)
    #         ## start adaptation flow matching
    #         ns_source = ns_source.to(self.device)
    #         ns_source_norm = self.normalize(ns_source, self.mean_adapt_ns, self.std_adapt_ns)
    #         traj_adaptation = torchdiffeq.odeint(
    #             lambda t, x: self.adaptation_model.forward(
    #                 x.float(),
    #                 torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32),
    #                 torch.cat([s, a], dim=-1).float()
    #             ),
    #             ns_source_norm.float(),
    #             torch.linspace(0, 1, self.config['adaptation_flow_steps'], device=self.device, dtype=torch.float32),
    #             atol=1e-4,
    #             rtol=1e-4,
    #             method="dopri5",
    #         )


    #         ns_adaptation = traj_adaptation[-1]

    #         ns_adaptation = self.inverse(ns_adaptation, self.mean_adapt_ns, self.std_adapt_ns).reshape(s.shape[0], n_samples, -1)

    #         gap = torch.mean(torch.norm(ns_adaptation - ns_source, dim=-1),dim=-1)
    #     if self.config['use_ema']:
    #         self.source_ema.restore()
    #         self.adaptation_ema.restore()

    #     return gap



            
        



        




