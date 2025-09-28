import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
from methods.meta_template import MetaTemplate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_distance(x1, x2, eps=1e-8):
    """ Robust cosine similarity for 3D or 4D tensors """
    try:
        if x1.dim() == 3 and x2.dim() == 3:
            dots = torch.matmul(x1, x2)
            norm1 = torch.norm(x1, dim=-1, keepdim=True)
            norm2 = torch.norm(x2, dim=-2, keepdim=True)
            scale = norm1 * norm2
        elif x1.dim() == 4 and x2.dim() == 4:
            dots = torch.matmul(x1, x2)
            norm1 = torch.norm(x1, dim=-1, keepdim=True)
            norm2 = torch.norm(x2, dim=-2, keepdim=True)
            scale = norm1 * norm2
        else:
            raise ValueError(f"Unsupported tensor dims in cosine_distance: {x1.shape}, {x2.shape}")
        return dots / (scale + eps)
    except Exception as e:
        print(f"cosine_distance error: {e}")
        return torch.zeros_like(dots)

class FewShotTransformer(MetaTemplate):
    def __init__(
        self,
        model_func,
        n_way,
        k_shot,
        n_query,
        variant="softmax",
        depth=1,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        initial_cov_weight=0.3,
        initial_var_weight=0.5,
        dynamic_weight=False,
    ):
        super().__init__(model_func, n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth

        dim = self.feat_dim

        self.current_accuracy = 0.0
        self.accuracy_threshold = 40.0
        self.use_advanced_attention = False

        self.ATTN = Attention(
            dim,
            heads,
            dim_head,
            variant,
            initial_cov_weight,
            initial_var_weight,
            dynamic_weight
        )

        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine" else nn.Linear(dim_head, 1)
        )

    def FFN_forward(self, x):
        return self.ffn(x)

    def final_layer_forward(self, x):
        return self.final_head(x)

    def update_accuracy(self, accuracy):
        flag = accuracy >= self.accuracy_threshold
        if flag != self.use_advanced_attention:
            self.use_advanced_attention = flag
            print(f"Switching to {'advanced' if flag else 'basic'} attention mechanism at accuracy: {accuracy:.2f}")

    def set_forward(self, x, is_feature=False):
        z_s, z_q = self.parse_feature(x, is_feature)
        z_s = z_s.view(self.n_way, self.k_shot, -1)
        z_proto = (z_s * self.sm(self.proto_weight)).sum(1).unsqueeze(0)
        z_q = z_q.view(self.n_way * self.n_query, -1).unsqueeze(1)

        h = z_proto
        for _ in range(self.depth):
            h = self.ATTN(h, z_q, z_q, use_advanced=self.use_advanced_attention) + h
            h = self.FFN_forward(h) + h

        return self.final_layer_forward(h).squeeze()

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).to(device)
        scores = self.set_forward(x)
        loss = self.loss_fn(scores, target)
        acc = (scores.argmax(1) == target).float().mean().item()
        self.update_accuracy(acc)
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight, initial_var_weight, dynamic_weight=False):
        super().__init__()
        self.heads = heads
        self.variant = variant
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_mlp = nn.Sequential(
                nn.LayerNorm(dim_head * 2),
                nn.Linear(dim_head * 2, dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 3),
                nn.Softmax(dim=-1),
            )
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        self.softmax = nn.Softmax(dim=-1)
        self.input_ln = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, dim_head * heads, bias=False)

        self.output_ln = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim_head * heads, dim)

        self.weight_history = []
        self.record_weights = False

    def variance_component(self, E, gamma, epsilon):
        sigma = torch.sqrt(torch.var(E, dim=1, keepdim=True) + epsilon)
        return torch.mean(torch.clamp(gamma - sigma, min=0.0))

    def covariance_component(self, E):
        B, S, D = E.shape
        E = E - E.mean(1, keepdim=True)
        cov = torch.matmul(E.transpose(1, 2), E) / max(S - 1, 1)
        offs = cov.flatten(start_dim=1)[:, :: D + 1].sum(-1)
        return (cov.square().sum((1, 2)) - offs).mean()

    def basic_attention_components(self, q, k):
        q_c = q - q.mean(-1, keepdim=True)
        k_c = k - k.mean(-1, keepdim=True)
        cov = torch.matmul(q_c, k_c.transpose(-1, -2)) / q.size(-1)
        q_var = torch.var(q, dim=-1, keepdim=True)
        k_var = torch.var(k, dim=-1, keepdim=True).transpose(-1, -2)
        var = torch.matmul(q_var, k_var) / q.size(-1)
        return cov, var

    def advanced_attention_components(self, f_q, f_k, gamma, epsilon):
        heads, batch_size, seq_q, dim = f_q.shape
        seq_k = f_k.shape[2]

        total_elements = f_k.numel()
        expected_elements = batch_size * heads * seq_k * dim

        if total_elements != expected_elements:
            seq_k = total_elements // (batch_size * heads * dim)
            if seq_k * batch_size * heads * dim != total_elements:
                raise RuntimeError(
                    f"Cannot resolve reshape, f_k has {total_elements} elements but shape mismatch"
                )

        f_q_reshaped = f_q.permute(1, 0, 2, 3).contiguous().view(batch_size * heads, seq_q, dim)
        f_k_reshaped = f_k.permute(1, 0, 2, 3).contiguous().view(batch_size * heads, seq_k, dim)

        f_q_mean = f_q_reshaped.mean(dim=-1, keepdim=True)
        f_k_mean = f_k_reshaped.mean(dim=-1, keepdim=True)

        f_q_centered = f_q_reshaped - f_q_mean
        f_k_centered = f_k_reshaped - f_k_mean

        cov_component = torch.bmm(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component /= (dim ** 0.5)

        f_q_var = torch.var(f_q_reshaped, dim=-1, keepdim=True)
        f_k_var = torch.var(f_k_reshaped, dim=-1, keepdim=True)

        var_component = torch.bmm(f_q_var, f_k_var.transpose(-1, -2))

        cov_component = cov_component.view(batch_size, heads, seq_q, seq_k)
        var_component = var_component.view(batch_size, heads, seq_q, seq_k)

        cov_component *= gamma
        cov_component += epsilon
        var_component *= gamma
        var_component += epsilon

        return cov_component, var_component

    def forward(self, q, k, v, use_advanced=False, gamma=1.0, epsilon=1e-5):
        qkv = self.to_qkv(torch.cat((q, k, v), dim=1))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = rearrange(q, "b n (h d) -> h b n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> h b n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> h b n d", h=self.heads)

        cosine_sim = cosine_distance(q, k.transpose(-1, -2))

        if self.variant == "cosine":
            try:
                if use_advanced:
                    cov_component, var_component = self.advanced_attention_components(q, k, gamma, epsilon)
                else:
                    cov_component, var_component = self.basic_attention_components(q, k)
            except Exception as e:
                print(f"Error in attention components: {e}, fallback to basic attention.")
                cov_component, var_component = self.basic_attention_components(q, k)

            if self.dynamic_weight:
                with torch.no_grad():
                    q_avg = q.mean(dim=(1, 2))
                    k_avg = k.mean(dim=(1, 2))
                    weights = self.weight_mlp(torch.cat([q_avg, k_avg], dim=-1))

                if self.record_weights and not self.training:
                    self.weight_history.append(weights.cpu().numpy().mean(axis=0))

                cos_w = weights[:, 0].view(-1, 1, 1, 1)
                cov_w = weights[:, 1].view(-1, 1, 1, 1)
                var_w = weights[:, 2].view(-1, 1, 1, 1)

                dots = cos_w * cosine_sim + cov_w * cov_component + var_w * var_component
            else:
                cov_w = torch.sigmoid(self.fixed_cov_weight)
                var_w = torch.sigmoid(self.fixed_var_weight)
                cos_w = 1.0 - cov_w - var_w
                dots = cos_w * cosine_sim + cov_w * cov_component + var_w * var_component
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attention = F.softmax(dots, dim=-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, "h b n d -> b n (h d)")
        return self.to_out(out)

