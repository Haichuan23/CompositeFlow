import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional



# def timestep_embedding(timesteps, dim, max_period=10000):
#     """Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period)
#         * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
#         / half
#     )
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half_dim = dim // 2
        # 确保 half_dim 至少为 1，以避免除以零或负数
        if half_dim <= 1:
             # 对于非常小的 dim，可能需要不同的处理方式，或者直接报错
             # 这里我们简单处理，避免 log(max_period) / 0 的情况
             # 或者可以考虑设置一个最小 half_dim
             # 例如: half_dim = max(1, dim // 2)
             # 但如果 dim=1, half_dim=0, 仍然有问题。
             # 更稳妥的是要求 dim >= 2
             assert dim >= 2, "Dimension must be >= 2 for SinusoidalPosEmb"
             emb = 0 # 或者其他合适的默认值/处理
        else:
            emb = math.log(max_period) / (half_dim - 1)

        # arange 的 end 参数是非包含的，所以如果 half_dim=1, arange(1) -> tensor([0.])
        freqs = torch.exp(-emb * torch.arange(half_dim).float())
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: shape [batch_size, 1] 或 [batch_size]
        输出 emb: shape [batch_size, dim]
        """
        freqs = self.freqs  # [half_dim]
        
        # 确保 x 是 2D 张量 [batch_size, 1] 以进行广播
        if x.ndim == 1:
            x = x.unsqueeze(-1) # 将 [batch_size] 变为 [batch_size, 1]
        
        # 保证x是float类型
        x = x.float() 

        # --- 修改核心 ---
        # 原代码: args = x[:, None] * freqs[None, :] # 输出 [batch, 1, half_dim]
        # 修改后: 利用广播机制 (batch, 1) * (1, half_dim) -> (batch, half_dim)
        args = x * freqs[None, :]  # 输出 [batch_size, half_dim]
        # -------------

        emb = torch.cat((args.sin(), args.cos()), dim=-1)  # 输出 [batch_size, dim]

        # 处理奇数维度 dim 的情况
        if self.dim % 2 == 1:
            # 创建一个形状为 [batch_size, 1] 的零张量，设备和类型与 emb 保持一致
            padding = torch.zeros_like(emb[:, :1]) 
            emb = torch.cat([emb, padding], dim=-1) # 输出 [batch_size, dim]
            
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #print(f"Inside ResidualBlock, before linear call:")
        #print(f"  x.shape: {x.shape}") # Input to the block
        activated_ln_x = self.activation(self.ln(x))
        #print(f"  activated_ln_x.shape: {activated_ln_x.shape}") # Input to the linear layer
        return x + self.linear(activated_ln_x)
        #return x + self.linear(self.activation(self.ln(x)))



class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #print(f"Inside ResidualMLP, before final_linear call:")
        #output_network = self.network(x)
        #print(f"  output_network.shape: {output_network.shape}") # Check output of the sequential part
        return self.final_linear(self.activation(self.network(x)))



class ResidualMLPGuidance(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = False,
            learned_sinusoidal_dim: int = 16,
            activation: str = "relu",
            layer_norm: bool = True,
            cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        if self.conditional:
            assert cond is not None
            #print('cond: ', cond.shape)
            #print('x: ', x.shape)
            x = torch.cat((x, cond), dim=-1)

        # time_embed = self.time_mlp(timesteps)
        # x = self.proj(x) + time_embed
        
        
        # print(f"Inside ResidualMLPGuidance, before residual_mlp call:")
        # print(f"  x.shape: {x.shape}") # x is the input to residual_mlp

        
        # res = self.residual_mlp(x)
        # return res
        # Inside ResidualMLPGuidance.forward
        # ... (after cat) ...
        time_embed = self.time_mlp(timesteps)
        #print(f"Shape before proj: x.shape={x.shape}") # Should be 2D after cat
        proj_x = self.proj(x)
        #print(f"Shape after proj: proj_x.shape={proj_x.shape}") # Should be 2D
        #print(f"Shape of time_embed: time_embed.shape={time_embed.shape}") # Should be 2D

        # === Problem likely here ===
        #print(f"DEBUG: About to add proj_x {proj_x.shape} and time_embed {time_embed.shape}")
        x = proj_x + time_embed
        #print(f"Shape after proj + time_embed: x.shape={x.shape}") # Check if this is 3D
        # ==========================

        res = self.residual_mlp(x)
        return res