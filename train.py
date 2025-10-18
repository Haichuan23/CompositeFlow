import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils # Assumed correct
import torch.nn as nn
import re
from pathlib import Path
# --- Make sure these imports point to your actual file locations ---
from algo.call_algo import call_algo # Assumed correct
from dataset.call_dataset import call_tar_dataset # Assumed correct
from envs.mujoco.call_mujoco_env import call_mujoco_env # Assumed correct
from envs.adroit.call_adroit_env import call_adroit_env # Assumed correct
from envs.antmaze.call_antmaze_env import call_antmaze_env # Assumed correct
from envs.infos import get_normalized_score # Assumed correct
# --- End of potentially custom imports ---
from tensorboardX import SummaryWriter
import wandb
from typing import Tuple, Union
import glob
import hashlib
import json

# --- Signal Handling Imports and Globals ---
import signal
import sys
# Global flag to indicate if a signal was received
graceful_exit_request = False

def signal_handler(signum, frame):
    """Sets the flag when a signal (like SIGTERM) is received."""
    global graceful_exit_request
    # Using print statements directly in handlers can sometimes be risky,
    # but for simple info messages it's often okay.
    print(f"\nINFO: Received signal {signal.strsignal(signum)}. Requesting graceful exit...\n", flush=True)
    graceful_exit_request = True
# --- End of Signal Handling Setup ---


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    """Evaluates the policy."""
    eval_env = env
    avg_reward = 0.
    avg_ep_len = 0.
    try:
        for episode_idx in range(eval_episodes):
            state, done = eval_env.reset(), False
            ep_reward = 0.
            ep_len = 0
            max_steps = getattr(eval_env, '_max_episode_steps', 1000) # Get max steps if available
            while not done:
                action = policy.select_action(np.array(state), test=True) # Use test=True for deterministic eval
                next_state, reward, done, info = eval_env.step(action)

                ep_reward += reward
                state = next_state
                ep_len += 1
                if ep_len >= max_steps: # Ensure termination if env doesn't handle it
                    done = True
            avg_reward += ep_reward
            avg_ep_len += ep_len
    except Exception as e:
        print(f"[Error] Exception during policy evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0 # Return 0 or handle error appropriately

    avg_reward /= eval_episodes
    avg_ep_len /= eval_episodes

    eval_id = f"Eval-{eval_cnt}" if eval_cnt is not None else "Evaluation"
    print(f"[{eval_id}] Avg Reward over {eval_episodes} episodes: {avg_reward:.3f} (Avg Ep Len: {avg_ep_len:.1f})")

    return avg_reward

# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs", help="Base directory for logs and models")
    parser.add_argument("--policy", default="SAC", help='Policy algorithm to use (e.g., SAC, VFlowPolicy)')
    parser.add_argument("--env", default="halfcheetah-friction", help="Environment name (e.g., hopper-friction, antmaze-medium-diverse-v0)")
    parser.add_argument('--srctype', default="medium", help='Dataset type for offline source domain (e.g., medium, expert)')
    parser.add_argument('--tartype', default="medium", help='Dataset type for offline target domain (e.g., random, medium)')
    parser.add_argument('--shift_level', default=0.5, help='Scale/type of dynamics shift')
    parser.add_argument('--mode', default=0, type=int, help='Training mode: 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument('--tar_env_interact_interval', default=10, type=int, help='Interaction frequency with target env (Modes 0, 1)')
    parser.add_argument('--max_step', default=int(4e5), type=int, help="Max *gradient steps*")
    # parser.add_argument('--max_step', default=int(4e5), type=int, help="Max *gradient steps*")
    parser.add_argument('--eval_freq', default=int(5e3), type=int, help="Evaluation frequency (gradient steps)")
    parser.add_argument('--params', default=None, help='JSON string for overriding config parameters')
    parser.add_argument('--load_model', default=None, help='Path prefix to load pre-trained model checkpoint') # Clarified purpose
    parser.add_argument('--resume_run_dir', default=None, help='Directory of a previous run to resume (loads latest checkpoint)') # Option to resume easily

    # VFlow / Pretrain specific args (can be overridden by --params or config file)
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for source model')
    parser.add_argument('--use_weight', action='store_true', help='Use importance weighting')
    parser.add_argument('--filter_percent', default=0.7, type=float, help='Filtering percentile for source samples')
    parser.add_argument('--start_gate_src_sample', default=100000.0, type=float, help='Step to start filtering/weighting')
    parser.add_argument('--beta', default=1.0, type=float, help='Temperature for weighting function')
    parser.add_argument('--dynamics_train_freq', default=5000, type=int, help='Dynamics model training frequency')
    ### CQL + pretrain flow
    parser.add_argument('--temperature_opt', action='store_true', help='Optimize SAC temperature alpha')
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--alpha', default=0.2, type=float, help='Initial SAC temperature alpha')
    parser.add_argument('--tar_cql', action='store_true', help='Use CQL on target domain dataset')
    parser.add_argument('--cql_alpha', default=10.0, type=float) # CQL specific args...
    parser.add_argument('--cql_max_target_backup', action='store_true')
    parser.add_argument('--cql_n_actions', default=10, type=int)
    parser.add_argument('--cql_temp', default=1.0, type=float)
    parser.add_argument('--cql_clip_diff_min', default=-1e6, type=float)
    parser.add_argument('--cql_clip_diff_max', default=1e6, type=float)
    parser.add_argument('--no_cql_importance_sample', action='store_false', dest='cql_importance_sample')
    parser.add_argument('--no_backup_entropy', action='store_false', dest='backup_entropy')
    parser.add_argument('--cql_lagrange', action='store_true')
    ## dynamics gap
    parser.add_argument('--dynamics_gap_reward_scale', default=0.0, type=float)
    parser.add_argument('--n_samples', default=100, type=int, help='Num samples for dynamics gap estimation')
    parser.add_argument('--downsample_src', default=1., type=float, help='Downsample ratio for source dataset loading')
    parser.add_argument('--upsample_src', action='store_true', help='Upsample source data during training')
    parser.add_argument('--use_sample_level', action='store_true', help='Use sample-level dynamics gap')
    parser.add_argument('--weight', default=2.5, type=float, help='Weight for BC loss in policy update (if applicable)') # Clarified purpose

    ## Extreme Target Shift 
    parser.add_argument("--extreme_shift", action = "store_true", help = "If set, use the extreme XML variants (broken limbs, etc.) in call_mujoco_env")

    args = parser.parse_args()

    # --- Add Signal Registration ---
    print("INFO: Registering SIGTERM handler for graceful shutdown.")
    signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGINT, signal_handler) # Optionally handle Ctrl+C
    # --- End of Signal Registration ---


    # --- Determine Domain and Env Names ---
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError(f"Domain not recognized for environment: {args.env}")
    print(f"Domain detected: {domain}")

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # Determine source and target environment configurations
    ref_env_name = args.env + '-' + str(args.shift_level)  # For normalization scores

    if domain == 'antmaze':
        # Antmaze names are often like 'antmaze-medium-play-v0'
        src_env_name_base = args.env.split('-')[0] + '-' + args.env.split('-')[1] # e.g., antmaze-medium
        src_env_config_name = src_env_name_base # Used for loading yaml config
        src_env_config_name = src_env_name_base + '-' + args.srctype 
        tar_env_name = args.env # Full name like antmaze-medium-play-v0
    elif domain == 'adroit':
        # Adroit names like 'pen-human-v0'
        src_env_name_base = args.env.split('-')[0] # e.g., pen
        src_env_config_name = src_env_name_base + '-' + args.srctype
        tar_env_name = args.env
    else: # mujoco
        # Mujoco names like 'hopper-friction'
        src_env_name_base = args.env.split('-')[0] # e.g., hopper
        src_env_config_name = src_env_name_base + '-' + args.srctype + '-v2'
        tar_env_name = args.env # Full name like hopper-friction

    print(f"Source Env Base: {src_env_name_base}, Target Env Name: {tar_env_name}, Config Name: {src_env_config_name}")

    # --- Create Environments ---
    src_env, src_eval_env = None, None
    tar_env, tar_eval_env = None, None
    src_d4rl_name = None # Initialize

    # Source Env/Dataset Setup
    if args.mode == 1 or args.mode == 3: # Offline source
        if domain == 'antmaze':
             src_d4rl_name = args.env # Assumes arg provides full D4RL name
        elif domain == 'adroit':
            src_d4rl_name = f"{src_env_name_base}-{args.srctype}-v0"
        else: # mujoco
             src_d4rl_name = f"{src_env_name_base}-{args.srctype}-v2"
        print(f"Using D4RL source dataset: {src_d4rl_name}")
        try:
            src_eval_env = gym.make(src_d4rl_name)
            src_eval_env.seed(args.seed + 100)
        except Exception as e:
            print(f"[Error] Could not make Gym environment for D4RL dataset {src_d4rl_name}: {e}")
            # Decide if you want to exit or maybe try to proceed if only dataset is needed
            src_eval_env = None # Ensure it's None
            # exit(1) # Optional: Exit if env is strictly needed
        src_env = None
    else: # Online source
        src_env_config = {'env_name': src_env_name_base, 'shift_level': args.shift_level}
        try:
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
            print(f"Using online source environment: {src_env_name_base} with shift {args.shift_level}")
        except Exception as e:
             print(f"[Error] Could not create online source environment {src_env_name_base}: {e}")
             exit(1)

    # Target Env/Dataset Setup
    tar_env_config = {'env_name': tar_env_name, 'shift_level': args.shift_level, "extreme_shift": args.extreme_shift}
    try:
        if args.mode == 2 or args.mode == 3: # Offline target
            print(f"Using offline target dataset: {tar_env_name} type {args.tartype}")
            # Need an eval env instance even if using offline data
            tar_eval_env = call_env[domain](tar_env_config)
            tar_eval_env.seed(args.seed + 100)
            tar_env = None # No online interaction with target
        else: # Online target
            print(f"Using online target environment: {tar_env_name} with shift {args.shift_level}, Extreme Shisft Status:{args.extreme_shift}")
            tar_env = call_env[domain](tar_env_config)
            tar_env.seed(args.seed)
            tar_eval_env = call_env[domain](tar_env_config)
            tar_eval_env.seed(args.seed + 100)
    except Exception as e:
            print(f"[Error] Could not create target environment/dataset {tar_env_name}: {e}")
            exit(1)


    if args.mode not in [0, 1, 2, 3]:
        raise NotImplementedError(f"Mode {args.mode} is not supported.")

    # --- Load Configuration ---
    policy_config_name = args.policy.lower()
    config_path = f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_base}.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded base config from: {config_path}")
    except FileNotFoundError:
        print(f"[Warning] Config file not found: {config_path}. Using default config values and args.")
        config = {}
    except Exception as e:
        print(f"[Error] Failed to load config file {config_path}: {e}")
        exit(1)

    # Override config with command-line JSON (--params)
    if args.params is not None:
        try:
            override_params = json.loads(args.params)
            config.update(override_params)
            print(f"Overridden config with --params: {args.params}")
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON in --params argument: {e}")
            exit(1)

    # Override config with specific command-line arguments (ensure these are present in args)
    override_keys = [
        'use_ema', 'use_weight', 'filter_percent', 'start_gate_src_sample', 'beta', 'dynamics_train_freq',
        'temperature_opt', 'actor_lr', 'critic_lr', 'alpha', 'tar_cql', 'cql_alpha', 'cql_max_target_backup',
        'cql_n_actions', 'cql_temp', 'cql_clip_diff_min', 'cql_clip_diff_max', 'cql_importance_sample',
        'backup_entropy', 'cql_lagrange', 'dynamics_gap_reward_scale', 'n_samples', 'downsample_src',
        'upsample_src', 'use_sample_level', 'weight', 'eval_freq', 
    ]
    for key in override_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    # --- Set Seeds ---
    print(f"Using seed: {args.seed}")
    if src_env: src_env.action_space.seed(args.seed)
    if tar_env: tar_env.action_space.seed(args.seed)
    # Use eval envs as they are guaranteed to be created if possible
    if src_eval_env: src_eval_env.action_space.seed(args.seed + 100)
    if tar_eval_env: tar_eval_env.action_space.seed(args.seed + 100)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # --- Env Dimensions and Action Limits ---
    # Use eval envs as they are guaranteed to exist
    if src_eval_env:
        state_dim = src_eval_env.observation_space.shape[0]
        action_dim = src_eval_env.action_space.shape[0]
        max_action = float(src_eval_env.action_space.high[0])
    elif tar_eval_env: # Fallback to target if no source eval env (e.g., only offline target?)
        state_dim = tar_eval_env.observation_space.shape[0]
        action_dim = tar_eval_env.action_space.shape[0]
        max_action = float(tar_eval_env.action_space.high[0])
    else:
        print("[Error] Could not determine state/action dimensions. No valid eval env.")
        exit(1)

    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}, Device: {device}")

    # --- Finalize Config ---
    config.update({
        'algo': args.policy.lower(),
        'env_name': args.env, # The original env name arg
        'task_name': src_env_config_name, # Base name used for config loading
        'env': src_env_config_name, 
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'shift_level': args.shift_level, # Keep original shift level value
        'extreme_shift': args.extreme_shift,
        'seed': args.seed,
        'mode': args.mode,
        'policy': args.policy,
        'device': device,
        # Extract everything after first '-' to handle both cases:
        # halfcheetah-morph-thigh -> morph-thigh
        # halfcheetah-gravity -> gravity
        'shift_type': src_env_config_name.split('-', 1)[1],
        # Add other args that might be needed by the policy constructor if not already covered
        'gamma': config.get('gamma', 0.99), # Ensure essential RL hparams are set
        'tau': config.get('tau', 0.005),
        'hidden_sizes': config.get('hidden_sizes', 256), # Example, policy might need this
        'batch_size': config.get('batch_size', 256),
        'update_interval': config.get('update_interval', 1), # For SAC style updates per step
    })

    # --- Create Output Directory and Logger ---
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device']) 
    config_str = json.dumps(config_to_save, sort_keys=True)
    # config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] # Slightly longer hash

    # Construct descriptive directory name
    run_name_parts = [args.policy, args.env]
    if args.mode == 1 or args.mode == 3: run_name_parts.append(f"src_{args.srctype}")
    if args.mode == 2 or args.mode == 3: run_name_parts.append(f"tar_{args.tartype}")
    run_name_parts.append(f"sl_{args.shift_level}")
    if args.extreme_shift:
        run_name_parts.append("extreme") 
    run_name_parts.append(f"s{args.seed}")
    run_name_parts.append(f"h{config_hash}")
    run_name = "-".join(run_name_parts)

    # Use the directory provided by --dir argument
    outdir = os.path.join(args.dir, run_name)
    # models_dir = os.path.join(outdir, "models")
    # buffers_dir = os.path.join(outdir, "buffers")
    tb_dir = os.path.join(outdir, "tb")

    print(f"Output directory: {outdir}")
    # os.makedirs(models_dir, exist_ok=True)
    # os.makedirs(buffers_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(tb_dir)

    # Save final config to the run directory
    config_save_path = os.path.join(outdir, 'final_config.json')
    try:
        # Convert device object to string for JSON serialization
        config_to_save = config.copy()
        config_to_save['device'] = str(config['device'])
        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, sort_keys=True)
        print(f"Saved final configuration to {config_save_path}")
    except Exception as e:
        print(f"[Warning] Could not save final configuration: {e}")

    # Log parameters to a text file
    try:
        with open(os.path.join(outdir, 'log.txt'), 'w') as f:
            f.write(f'Run Name: {run_name}\n')
            f.write(f'Command: python {" ".join(sys.argv)}\n\n') # Record command
            f.write(f'Policy: {args.policy}; Env: {args.env}; Seed: {args.seed}; Mode: {args.mode}\n\n')
            f.write('--- Configuration ---\n')
            # Sort config items for consistent logging
            for key, value in sorted(config.items()):
                 # Convert device object to string for logging
                 log_value = str(value) if isinstance(value, torch.device) else value
                 f.write(f'{key}: {log_value}\n')
            f.write('\n--- Args ---\n')
            # Sort args for consistent logging
            for key, value in sorted(vars(args).items()):
                 f.write(f'{key}: {value}\n')
    except Exception as e:
        print(f"[Warning] Could not write log.txt file: {e}")


    # --- Initialize Policy ---
    try:
        policy = call_algo(args.policy, config, args.mode, device) # Pass full config
    except Exception as e:
        print(f"[Error] Failed to initialize policy '{args.policy}': {e}")
        exit(1)

    # --- Initialize Replay Buffers ---
    buffer_size = config.get('buffer_size', int(1e6))
    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=buffer_size)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=buffer_size)

    # --- Resuming Logic ---
    start_step = 0 # Start from step 0 by default

    print("Starting training from scratch.")

    # --- Weights & Biases Initialization ---
    # Initialize wandb to None first, attempt to init, and proceed if it fails
    wandb_instance = None
    try:
        wandb_run_name = run_name # Use the descriptive run name generated earlier
        if hasattr(policy, 'total_it') and policy.total_it > 0: # If loaded a checkpoint
             wandb_run_name = f"resume_{run_name}"
        if args.extreme_shift:
            group = f"{args.env}-{args.srctype}-{args.shift_level}-extreme-True"
        else:
            group = f"{args.env}-{args.srctype}-{args.shift_level}"
        wandb_instance = wandb.init(
            project="Flow_RL", # Replace with your project name
            entity=None, # Replace with your wandb entity (username or team) or leave as None for default
            name=wandb_run_name,
            config=config, # Log the final config
            dir=outdir, # Save wandb files within the run directory
            resume="allow", # Allow resuming if run_id matches
            id=run_name, # Use base run name as ID for potential resume
            group=group,
        )
        
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"[Error] Failed to initialize Weights & Biases: {e}")
        print("[Warning] Wandb logging disabled.")
        # wandb_instance remains None


    # --- Load Offline Datasets ---
    # Source dataset (Modes 1, 3)
    if args.mode == 1:
        # Ensure src_eval_env exists and src_d4rl_name is set
        if src_eval_env and src_d4rl_name:
            try:
                print(f"Loading source D4RL dataset: {src_d4rl_name}")
                d4rl_dataset = d4rl.qlearning_dataset(src_eval_env)
                # Load data into the buffer (already on the correct device)
                src_replay_buffer.convert_D4RL(d4rl_dataset)
                if 'antmaze' in args.env:
                    print("Adjusting Antmaze rewards (-1.0)")
                    src_replay_buffer.reward -= 1.0
                print(f"Loaded {src_replay_buffer.size} transitions into source buffer.")
                # Optional downsampling
                if args.downsample_src < 1.0:
                     src_replay_buffer.downsample(args.downsample_src)
                     print(f"Downsampled source buffer to {src_replay_buffer.size} transitions.")
            except Exception as e:
                 print(f"[Error] Failed to load or process D4RL source dataset {src_d4rl_name}: {e}")
                 # exit(1) # Decide whether to exit
        else:
            print("[Error] Source eval env or D4RL name not available for offline loading (Mode 1 or 3).")
            # exit(1)
    else:
        print(f"[Error] Invalid training mode specified: {args.mode}")

    # --- Initial Policy Evaluation ---
    eval_cnt = 0
    # Only evaluate if not resuming from a later step, or always evaluate?
    # Let's always evaluate the current policy state
    print("\n--- Initial Evaluation (Before Training Loop Starts/Resumes) ---")
    if src_eval_env:
        init_src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=f"InitSrc-{start_step}")
        writer.add_scalar('eval/source_return', init_src_eval_return, global_step=start_step)
        if wandb_instance: wandb_instance.log({'eval/source_return': init_src_eval_return}, step=start_step)
    # eval_cnt += 1 # Not really used later, maybe remove
    if tar_eval_env:
        init_tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=f"InitTar-{start_step}")
        init_eval_normalized_score = get_normalized_score(init_tar_eval_return, ref_env_name)
        writer.add_scalar('eval/target_return', init_tar_eval_return, global_step=start_step)
        writer.add_scalar('eval/target_normalized_score', init_eval_normalized_score, global_step=start_step)
        if wandb_instance: wandb_instance.log({
            'eval/target_return': init_tar_eval_return,
            'eval/target_normalized_score': init_eval_normalized_score
        }, step=start_step)
    # eval_cnt += 1
    print("-------------------------------------------------------------\n")


    # --- Training Loop ---
    max_steps = args.max_step
    eval_freq = int(config['eval_freq'])
    batch_size = int(config['batch_size'])

    print(f"Starting training loop from step {start_step} up to {max_steps}")
    print(f"Using Mode: {args.mode}")

    # ================== MODE 1: Offline-Online ==================
    if args.mode == 1:
        if not tar_env:
            print("[Error] Mode 1 requires an online target environment.")
            exit(1)
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        print(f"Starting/Resuming Mode 1 training loop from step {start_step}...")
        for t in range(start_step, max_steps):

            # --- CHECK FOR EXIT SIGNAL ---
            if graceful_exit_request:
                print(f"INFO: Graceful exit requested at step {t}. Saving final state (Mode 1)...")
                print("INFO: --save-model not specified, skipping final save.")
                print("INFO: Closing resources...")
                writer.close()
                if wandb_instance: wandb_instance.finish(exit_code=99)
                print("INFO: Exiting gracefully with code 99.")
                sys.exit(99)
            # --- END OF EXIT SIGNAL CHECK ---

             # --- Target Env Interaction (Periodic based on gradient steps) ---
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)
                tar_next_state, tar_reward, tar_done, tar_info = tar_env.step(tar_action)
                tar_real_done = tar_done and not tar_info.get('TimeLimit.truncated', False)
                tar_done_bool = float(tar_real_done)
                if 'antmaze' in args.env: tar_reward -= 1.0
                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)
                tar_state = tar_next_state
                tar_episode_reward += tar_reward

                if tar_done:
                    print(f"Step: {t+1}/{max_steps} | Tar Ep Num: {tar_episode_num+1} | Ep Steps: {tar_episode_timesteps} | Reward: {tar_episode_reward:.2f}")
                    train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                    writer.add_scalar('train/target_return', tar_episode_reward, global_step=t+1)
                    writer.add_scalar('train/target_normalized_score', train_normalized_score, global_step=t+1)
                    if wandb_instance: wandb_instance.log({
                        'train/target_return': tar_episode_reward,
                        'train/target_normalized_score': train_normalized_score,
                    }, step=t+1)
                    tar_state, tar_done = tar_env.reset(), False
                    tar_episode_reward = 0
                    tar_episode_timesteps = 0
                    tar_episode_num += 1

            # --- Policy Training ---
            # Train if either buffer has enough samples (src is static, tar fills up)
           
            policy.train(src_replay_buffer, tar_replay_buffer, batch_size, writer)

            # --- Evaluation and Periodic Checkpointing ---
            if (t + 1) % eval_freq == 0:
                if tar_eval_env:
                    tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=f"Step{t+1}-Tar")
                    eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                    writer.add_scalar('eval/target_return', tar_eval_return, global_step=t+1)
                    writer.add_scalar('eval/target_normalized_score', eval_normalized_score, global_step=t+1)
                    if wandb_instance: wandb_instance.log({
                        'test/target_return': tar_eval_return,
                        'test/target_normalized_score': eval_normalized_score,
                    }, step=t+1)    

                if (t + 1) == 400000:
                    tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=f"Step{t+1}-Tar")
                    eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                    if wandb_instance: wandb_instance.log({
                        'test/target_return_400K': tar_eval_return,
                        'test/target_normalized_score_400K': eval_normalized_score,
                    })    
    else:
         print(f"[Error] Invalid training mode specified: {args.mode}")


    # --- Normal Finish ---
    print("="*40)
    print(f"Training loop finished normally after {max_steps} steps.")
    print("="*40)
    writer.close()
    if wandb_instance:
        wandb_instance.finish() # Normal exit code 0