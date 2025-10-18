# Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data

<p align="center">
  <br />
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <a href="Python 3.8"><img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg" /></a>
</p>

![A brief overview of the Composite Flow Framework.](./imgs/ODRLbenchmark.png)
🔥 [Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](https://arxiv.org/abs/2505.23062)

CompositeFlow adapts RL agents to dynamics shift by composing flow matching models for source and target MDPs and guiding exploration with learned dynamics gaps.

## 🛠️ Installation (CompositeFlow)

Clone project and create environment with conda:

```bash
# Create and activate conda environment
conda create -n compositeflow python=3.8
conda activate compositeflow

# Install dependencies
conda env update --file environment.yml --prune

## 🚀 Run Training (CompositeFlow)

After setting up the environment, you can launch a single training run with:

```bash
python train.py \
    --policy vflow \
    --env hopper-friction \
    --mode 1 \
    --srctype medium-replay \
    --shift_level 5.0 \
    --seed 0 \
    --n_samples 30 \
    --dynamics_gap_reward_scale 0.1 \
    --filter_percent 0.8 \
    --dir training_output/seed_0

## 🧩 Training (Baselines)

To train a single baseline agent (e.g., BC-SAC) on Hopper-Friction:

```bash
python train.py \
    --policy bc_sac \
    --env hopper-friction \
    --mode 1 \
    --srctype medium-replay \
    --shift_level 5.0 \
    --seed 0 \
    --dir training_output/bc_sac_seed_0

## Licences

Our repository is licensed under the MIT licence. The adopted ODRL codeabse, Gym environments, and mujoco-py are also licensed under the MIT License. For the D4RL library (including the Antmaze domain and the Adroit domain, and offline datasets), all datasets are licensed under the Creative Commons Attribution 4.0 License (CC BY), and code is licensed under the Apache 2.0 License.

## References
This codebase has been adapted from [ORDL](https://arxiv.org/html/2410.20750v1). 

## 📄Citing CompositeFlow
Please consider citing us if you find our work useful!

```
@article{kong2025composite,
  title={Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data},
  author={Kong, Lingkai and Wang, Haichuan and Wang, Tonghan and Xiong, Guojun and Tambe, Milind},
  journal={arXiv preprint arXiv:2505.23062},
  year={2025},
  url={https://arxiv.org/abs/2505.23062}
}
```
