import sys
sys.path.append(".")
import torch
import torch.nn as nn
from utils.util import *
from functools import partial
import math
from torch.nn import Parameter

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat,) + hid_feat + (out_feat,)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_I(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.I_qk = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.I_conv = nn.Linear(head_dim, 15, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, interact_feature, mask=None):
        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads

        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4)  # [3, B, #heads, N, C//#heads]
        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]  # [B, #heads, N, C//#heads]
        attn_J = (J_q @ J_k.transpose(-2, -1))  # [B, #heads, N, N]

        I_qkv = self.I_qk(interact_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4)
        I_q, I_k, I_v = I_qkv[0], I_qkv[1], I_qkv[2]  # [B, #heads, N, C//#heads]
        attn_I = (I_q @ I_k.transpose(-2, -1))  # [B, #heads, N, N]
        attn_I_linear = self.I_conv(I_v)  # [B, #heads, N, N]

        attn = (attn_J + attn_I + attn_I_linear) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, #heads, N, N]

        x = (attn @ J_v).transpose(1, 2).reshape(B, N, -1)  # [B, N, -1]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # dim = 128， num_heads = 8
        super().__init__()
        self.attn_i = Attention_I(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm_attn1 = norm_layer(dim)
        self.norm_attn2 = norm_layer(dim)
        self.norm_joint = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_joint = Mlp(in_features=dim, hidden_features=dim//2, act_layer=act_layer, drop=drop)

    def forward(self, joint_feature, relation_feature, mask=None):
        ## joint feature update through attention mechanism
        joint_feature = joint_feature + self.drop_path(self.attn_i(self.norm_attn1(joint_feature), self.norm_attn2(relation_feature), mask))
        joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))  # joint_feature (4,45,128)
        return joint_feature

class Block2(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm_attn = norm_layer(dim)
        self.norm_joint = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_joint = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, joint_feature, mask=None):
        joint_feature = joint_feature + self.drop_path(self.attn(self.norm_attn(joint_feature), mask))
        joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))
        return joint_feature


class JRTransformer(nn.Module):
    def __init__(self, in_joint_size=50 * 9, feat_size=128, out_joint_size=25 * 3, num_heads=16, depth=6,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.joint_encoder = MLP(in_joint_size, feat_size, (512, 1024))
        # There is interaction.
        self.attn_encoder = nn.ModuleList([
            Block(feat_size, num_heads, qkv_bias=True, qk_scale=0.6, norm_layer=norm_layer, drop_path=0.1)
            for i in range(depth)])
        # There is no interaction.
        self.attn_encoder3 = nn.ModuleList([
            Block2(feat_size, num_heads, qkv_bias=True, qk_scale=0.6, norm_layer=norm_layer, drop_path=0.1)
            for i in range(depth)])

        self.joint_decoder = MLP(feat_size, out_joint_size)
        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_joint, y_joint):
        B, J, T, D = x_joint.shape
        if torch.all(y_joint == 0):
            # There is no interaction.
            x_joint = x_joint.reshape(B, J, -1)
            feat_x_joint = self.joint_encoder(x_joint.cuda())
            pred_aux_joint = [self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3)]
            for i in range(len(self.attn_encoder3)):
                blk = self.attn_encoder3[i]
                feat_x_joint = blk(feat_x_joint)  # blk
                pred_aux_joint.append(self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3))
            pred = self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3)
            return pred, pred_aux_joint
        else:
            # There is interaction.
            in_factor_ = []
            for m in range(0, y_joint.shape[1] - 1, J):
                if torch.all(y_joint[:, m:m + J] == 0):
                    continue
                else:
                    interact_person = y_joint[:, m:m + J]  # B,J,T,9
                    dis_ = distance_ab(x_joint[..., :3], interact_person[..., :3])  # B,J,T relative distance
                    vel_ = vel_ab(x_joint[..., 3:6], interact_person[..., 3:6])  # B,J,T,3  relative velocity
                    acc_ = acc_ab(x_joint[..., 3:6], interact_person[..., 3:6])  # B,J,T,3  relative acceleration
                    in_factor = torch.cat((dis_.unsqueeze(-1), vel_, acc_), dim=-1).reshape(B,J,-1)  # B,J,7T
                    in_factor_.append(in_factor)
            compound = torch.cat(in_factor_, dim=1)  # B,nJ,7T
            compound = compound.reshape(B, J, -1, compound.shape[-1]).reshape(B, J, -1)  # B,J,7nT

            x_joint = x_joint.reshape(B, J, -1)
            feat_x_joint = self.joint_encoder(x_joint.cuda())
            pred_aux_joint = [self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3)]

            message_encoder = MLP(compound.shape[-1], feat_x_joint.shape[-1], (512, 512)).cuda()
            feat_compound = message_encoder(compound)  # B,J,512

            for i in range(len(self.attn_encoder)):
                blk = self.attn_encoder[i]
                blk2 = self.attn_encoder3[i]
                if i == 0:
                    feat_x_joint = blk(feat_x_joint, feat_compound)
                else:
                    feat_x_joint = blk2(feat_x_joint)
                pred_aux_joint.append(self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3))
            pred = self.joint_decoder(feat_x_joint).contiguous().view(B, J, -1, 3)
            return pred, pred_aux_joint
