import torch.nn.functional as F
from torch import nn, Tensor
from typing import Literal
import torch
# from jaxtyping import Array, Float32

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p2 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p3 = nn.Linear(in_features=D*self.M, out_features=D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    def __init__(self, D: int, kernel_size: int=4):
        super().__init__()
        # A separable 1D convolution:
        # - Input channels = output channels = D
        # - groups = D makes it depthwise (channel-wise) convolution.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            bias=False,
            padding=kernel_size // 2  # optional, to preserve sequence length
        )

    def forward(self, x: Tensor) -> Tensor:
        # https://chatgpt.com/share/67692a55-5224-8005-a271-80067aa3bcbb
        # B = Batch size
        # T = Sequence length
        # D = Feature dimension
        # x: (B, T, D)
        # Transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)  # (B, D, T)
        # Transpose back to (B, T, D)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.): #
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5  # 0.177
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, mask=None):
        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads

        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4)  # [3, B, #heads, N, C//#heads]
        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]  # [B, #heads, N, C//#heads]
        attn = (J_q @ J_k.transpose(-2, -1))  # [B, #heads, N, N]
        attn = attn * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, #heads, N, N]
        x = (attn @ J_v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        x = x + joint_feature
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class Attention_se(nn.Module):
#     def __init__(self, dim, proj_drop=0.):
#         super().__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(dim, dim // 16, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(dim // 16, dim, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, joint_feature):
#         j_x = joint_feature.permute(0, 2, 1).unsqueeze(2)
#         j_se = self.se(j_x)  # B,C,N,N -> B,C,1,1
#         j_se = j_x * j_se.expand_as(j_x)  # B,C,1,N
#         j_se = j_se.squeeze(2).permute(0, 2, 1)  # B,C,N-> B,N,C
#
#         x = j_se + joint_feature
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

def check_for_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values!")

class Recurrent_block(nn.Module):
    def __init__(self, D:int, D_rnn:int=...,
                 approximate:Literal['none', 'tanh']='none'):
        super().__init__()
        self.D = D
        # self.D_rnn = D_rnn
        # self.gelu = nn.GELU(approximate)
        # self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        # self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        # self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        # self.separableConv1D = Temporal_Conv1D(D_rnn, kernel_size=3)
        self.attn = Attention(self.D, num_heads=16, qkv_bias=True, qk_scale=0.6, proj_drop=0.2)


    def forward(self, x:Tensor) -> Tensor:
        # x: 30,15,512
        # # left branch
        # x1 = self.p1(x)
        # x1 = self.gelu(x1)

        # right branch
        # x2 = self.p2(x)
        # x2 = self.separableConv1D(x2)
        x2 = self.attn(x)  # x2  self.attn隐藏层为D_rnn

        # y = x1 * x2  # element-wise multiplication
        # y = self.p3(x2)
        return x2


class Residual_block(nn.Module):
    def __init__(self, D:int):
        super().__init__()
        self.mlp = Gated_MLP_block(D, expansion_factor=3)
        self.tmb = Recurrent_block(D, D_rnn=int(2*D))
        self.rmsnorm = RMSNorm(d=D) # ?

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.rmsnorm(x)
        x1 = self.tmb(x1)
        y1 = x + x1

        # x2 = self.rmsnorm(y1)
        # x2 = self.mlp(x2)
        #
        # y2 = y1 + x2

        return y1