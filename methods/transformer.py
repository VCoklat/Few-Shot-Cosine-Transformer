import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
import pdb
import IPython
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InterventionTracker:
    """Smart intervention tracking to prevent spam and detect real issues"""
    def __init__(self):
        self.variance_history = []
        self.intervention_cooldown = 0
        self.last_major_intervention = 0
        self.stuck_counter = 0

    def should_intervene(self, current_variance, batch_idx):
        """Decide if intervention is needed based on trends"""
        # Always track variance
        self.variance_history.append(current_variance)
        if len(self.variance_history) > 20:  # Keep last 20 measurements
            self.variance_history.pop(0)

        # Reduce cooldown
        if self.intervention_cooldown > 0:
            self.intervention_cooldown -= 1
            return False, "cooling_down"

        # Check if we're truly stuck (not just low variance)
        if len(self.variance_history) >= 10:
            recent_vars = self.variance_history[-10:]
            variance_trend = max(recent_vars) - min(recent_vars)
            mean_variance = np.mean(recent_vars)

            # Stuck detection: low variance AND no improvement
            if mean_variance < 0.001 and variance_trend < 0.0005:
                self.stuck_counter += 1
                if self.stuck_counter >= 5:  # Confirmed stuck
                    self.stuck_counter = 0
                    self.intervention_cooldown = 15  # Longer cooldown for major intervention
                    return True, "truly_stuck"
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # Quick intervention for extremely low variance
        if current_variance < 0.0001:
            self.intervention_cooldown = 5
            return True, "emergency"

        return False, "normal"

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 initial_cov_weight=0.01, initial_var_weight=0.01, dynamic_weight=True,
                 gamma=0.01, lambda_reg=0.001):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        # SUPER AGGRESSIVE: Enhanced intervention system
        self.intervention_tracker = InterventionTracker()
        self.training_epoch = 0
        self.batch_count = 0
        self.major_interventions = 0
        self.last_variance = 0.0

        # Keep all innovations with SUPER AGGRESSIVE settings
        self.ATTN = SuperAggressiveAttention(dim, heads=heads, dim_head=dim_head, variant=variant,
                                           initial_cov_weight=initial_cov_weight,
                                           initial_var_weight=initial_var_weight,
                                           dynamic_weight=dynamic_weight,
                                           gamma=gamma,
                                           lambda_reg=lambda_reg)

        # SUPER AGGRESSIVE: Much more flexible temperature
        self.sm = nn.Softmax(dim=-2)
        self.temperature_sm = nn.Parameter(torch.ones(1) * 3.0)  # Start lower

        # SUPER AGGRESSIVE: Larger initialization for diversity
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.2)  # Much larger

        # SUPER AGGRESSIVE: Enhanced FFN with more diversity features
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),  # Higher dropout
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.3)
        )

        # Enhanced classification
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.2),
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dim_head, 1)
            )

    def super_aggressive_intervention(self, current_variance, intervention_type):
        """SUPER AGGRESSIVE interventions that actually work"""

        if intervention_type == "emergency":
            self.major_interventions += 1
            print(f"\n🚨 EMERGENCY INTERVENTION #{self.major_interventions}")
            print(f"   Variance: {current_variance:.8f} (CRITICAL)")

            with torch.no_grad():
                # MASSIVE noise injection
                proto_noise_scale = 0.5  # HUGE
                proto_noise = torch.randn_like(self.proto_weight) * proto_noise_scale
                self.proto_weight.data.add_(proto_noise)

                # RESET temperature dramatically
                self.temperature_sm.data.fill_(25.0)  # VERY high

                # Force dynamic weight reset
                self.ATTN.nuclear_weight_reset()

                print(f"   💥 MASSIVE proto noise: {proto_noise_scale}")
                print(f"   🌡️  EXTREME temp reset: 25.0")
                print(f"   ☢️  NUCLEAR weight reset applied")

            return True

        elif intervention_type == "truly_stuck":
            self.major_interventions += 1
            print(f"\n⚡ MAJOR STUCK-BREAKING INTERVENTION #{self.major_interventions}")
            print(f"   Variance: {current_variance:.8f} (STUCK)")

            with torch.no_grad():
                # Progressive noise injection
                proto_noise_scale = 0.3
                proto_noise = torch.randn_like(self.proto_weight) * proto_noise_scale
                self.proto_weight.data.add_(proto_noise)

                # Smart temperature management
                current_temp = self.temperature_sm.item()
                if current_temp < 20.0:
                    new_temp = min(current_temp * 2.0, 20.0)
                    self.temperature_sm.data.fill_(new_temp)
                    print(f"   🌡️  Temperature DOUBLED: {current_temp:.2f} → {new_temp:.2f}")
                else:
                    # If temp already high, do parameter noise instead
                    self.ATTN.emergency_parameter_noise()
                    print(f"   💥 Alternative: parameter noise injection")

                # Force break dynamic weight symmetry
                self.ATTN.force_weight_diversity()
                print(f"   🎛️  Forced weight diversity")

            return True

        return False

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # SUPER AGGRESSIVE: Much lighter normalization for diversity
        norm_strength = 0.6  # Even lighter
        z_support = F.normalize(z_support, p=2, dim=-1) * norm_strength + z_support * (1 - norm_strength)
        z_query = F.normalize(z_query, p=2, dim=-1) * norm_strength + z_query * (1 - norm_strength)

        # Dynamic temperature with much wider range
        current_temp = torch.clamp(self.temperature_sm, min=0.1, max=100.0)

        # Prototype computation with SUPER AGGRESSIVE diversity enforcement
        proto_logits = self.proto_weight / current_temp
        proto_weights = self.sm(proto_logits)

        # SUPER AGGRESSIVE: Much more uniform weighting
        uniform_weight = 0.2  # Higher uniform component
        proto_weights = proto_weights * (1 - uniform_weight) + uniform_weight / self.k_shot

        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)

        x, query = z_proto, z_query

        # Enhanced attention with SUPER AGGRESSIVE diversity preservation
        for layer_idx in range(self.depth):
            x_input = x.clone()
            attn_out = self.ATTN(q=x, k=query, v=query)

            # SUPER AGGRESSIVE: Variance-adaptive residual scaling
            if self.training:
                # Much stronger residual when variance is low
                variance_boost = max(1.0, 20.0 - self.last_variance * 5000)  # Much more aggressive
                scale_factor = 0.1 * min(variance_boost, 3.0)  # Cap at 3x
            else:
                scale_factor = 0.1

            x = x_input + scale_factor * attn_out

            # Enhanced FFN
            ffn_out = self.FFN(x.view(-1, x.size(-1))).view(x.shape)
            x = x + 0.2 * ffn_out  # Stronger FFN residual

        # SUPER AGGRESSIVE: More permissive final processing
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=0.01, max=50.0)  # Much wider range

        scores = self.linear(x).squeeze()

        # SUPER AGGRESSIVE: Emergency diversity injection
        if self.training:
            score_std = torch.std(scores)
            if score_std < 0.05:  # Higher threshold for emergency
                noise_scale = max(0.2, 0.2 / (score_std + 1e-8))  # Much larger noise
                noise = torch.randn_like(scores) * min(noise_scale, 2.0)  # Cap noise
                scores = scores + noise
                if score_std < 0.01:  # Really critical
                    print(f"🆘 EMERGENCY SCORE DIVERSITY: std={score_std:.6f}, noise={noise_scale:.3f}")

        # SUPER AGGRESSIVE: Adaptive temperature based on score health
        score_range = scores.max() - scores.min()
        if score_range < 0.2:  # Scores too compressed
            temperature = 0.05  # Make VERY sensitive
        elif score_range > 5.0:  # Scores too spread
            temperature = 2.0   # Make less sensitive
        else:
            temperature = 1.0

        scores = scores / temperature
        return scores

    def set_forward_loss(self, x):
        """Enhanced loss with SUPER AGGRESSIVE diversity preservation - TENSOR FIXED"""
        self.batch_count += 1
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)
        current_variance = torch.var(scores).item()
        self.last_variance = current_variance

        # SUPER AGGRESSIVE: Smart intervention system
        should_intervene, intervention_type = self.intervention_tracker.should_intervene(
            current_variance, self.batch_count)

        if should_intervene:
            intervention_applied = self.super_aggressive_intervention(current_variance, intervention_type)
            if intervention_applied:
                # Re-compute scores after intervention
                scores = self.set_forward(x)
                new_variance = torch.var(scores).item()
                print(f"   📊 Variance change: {current_variance:.6f} → {new_variance:.6f}")
                current_variance = new_variance

        # Enhanced base loss
        base_loss = self.loss_fn(scores, target)

        # SUPER AGGRESSIVE: Adaptive label smoothing
        if self.training:
            # Much more aggressive smoothing when variance is low
            if current_variance < 0.001:
                smooth_factor = 0.4  # Very high smoothing
            elif current_variance < 0.01:
                smooth_factor = 0.2  # Medium smoothing
            else:
                smooth_factor = 0.1  # Light smoothing

            # Create smoothed targets
            num_classes = scores.size(1)
            smooth_target = torch.full_like(scores, smooth_factor / num_classes)
            batch_indices = torch.arange(target.size(0), device=target.device)
            smooth_target[batch_indices, target] += (1.0 - smooth_factor)

            log_probs = F.log_softmax(scores, dim=1)
            smooth_loss = -torch.sum(log_probs * smooth_target, dim=1).mean()

            total_loss = 0.6 * base_loss + 0.4 * smooth_loss
        else:
            total_loss = base_loss

        # SUPER AGGRESSIVE: Enhanced diversity regularization
        if self.training:
            probs = F.softmax(scores, dim=1)

            # 1. Class distribution regularization
            class_probs = probs.mean(dim=0)
            uniform_target = torch.ones_like(class_probs) / self.n_way
            diversity_loss = F.kl_div(torch.log(class_probs + 1e-8), uniform_target, reduction='sum')

            # 2. Strong entropy bonus
            score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy_bonus = 0.2 * score_entropy  # Much stronger

            # 3. Variance preservation loss - FIXED TENSOR VERSION
            variance_tensor = torch.var(scores)
            if variance_tensor < 0.001:  # Very low variance
                variance_loss = -2.0 * torch.log(variance_tensor + 1e-8)  # Strong penalty
            else:
                variance_loss = -0.5 * torch.log(variance_tensor + 1e-8)  # Light penalty

            total_loss = total_loss + 0.02 * diversity_loss + entropy_bonus + 0.001 * variance_loss

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)

        return acc, total_loss

class SuperAggressiveAttention(nn.Module):
    """SUPER AGGRESSIVE attention with all your innovations + emergency systems"""
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.01, 
                 initial_var_weight=0.01, dynamic_weight=True, gamma=0.01, lambda_reg=0.001):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        # SUPER AGGRESSIVE: Enhanced dynamic weighting
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Dropout(0.4),  # High dropout
                nn.Linear(dim_head, dim_head // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(dim_head // 2, 3),
                nn.Softmax(dim=-1)
            )
            self.weight_temperature = nn.Parameter(torch.ones(1) * 1.0)  # Start low
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # SUPER AGGRESSIVE: Enhanced processing
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.2)
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.2)
        ) if project_out else nn.Identity()

        self.weight_history = []
        self.record_weights = False

    def nuclear_weight_reset(self):
        """NUCLEAR OPTION: Complete weight reset"""
        if self.dynamic_weight:
            with torch.no_grad():
                for param in self.weight_predictor.parameters():
                    if param.dim() > 1:
                        # Reinitialize with Xavier
                        nn.init.xavier_uniform_(param)
                    else:
                        param.fill_(0.01)

                # Reset temperature
                self.weight_temperature.data.fill_(0.5)
                print(f"☢️  NUCLEAR: Complete weight predictor reset")

    def emergency_parameter_noise(self):
        """Inject noise into all parameters"""
        if self.dynamic_weight:
            with torch.no_grad():
                for param in self.weight_predictor.parameters():
                    noise = torch.randn_like(param) * 0.1 * param.std()
                    param.add_(noise)
                print(f"💥 EMERGENCY: Parameter noise injection")

    def force_weight_diversity(self):
        """Force weights to be diverse"""
        if self.dynamic_weight:
            with torch.no_grad():
                # Add asymmetric bias to break symmetry
                if len(list(self.weight_predictor.parameters())) > 0:
                    final_layer = list(self.weight_predictor.parameters())[-2]  # Final linear layer weights
                    if final_layer.dim() == 2 and final_layer.size(1) == 3:
                        # Add different bias to each output
                        bias = torch.tensor([0.1, -0.05, -0.05], device=final_layer.device)
                        final_layer.data += bias.unsqueeze(0).expand_as(final_layer) * 0.1
                        print(f"🎛️  FORCED: Asymmetric weight bias applied")

    def compute_regularized_covariance(self, f_q, f_k):
        """Your covariance formula with SUPER AGGRESSIVE stability"""
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # Enhanced stability
        E_bar = f_k.mean(dim=2, keepdim=True)
        f_k_centered = f_k - E_bar
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)

        # Compute covariance with adaptive regularization
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5 + 1e-6)

        # SUPER AGGRESSIVE: Adaptive regularization based on component health
        base_reg = self.lambda_reg / max(m, 1)
        component_std = torch.std(cov_component)
        if component_std < 1e-6:  # Component is dead
            regularization_factor = base_reg * 10  # Much stronger regularization
        else:
            regularization_factor = base_reg

        cov_component = regularization_factor * cov_component
        cov_component = torch.clamp(cov_component, -50.0, 50.0)  # Wider range

        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """Your variance formula with SUPER AGGRESSIVE diversity preservation"""
        # Lighter normalization
        f_q_norm = F.normalize(f_q, p=2, dim=-1, eps=1e-6)
        f_k_norm = F.normalize(f_k, p=2, dim=-1, eps=1e-6)

        # Cosine similarity with wider range
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))
        cosine_sim = torch.clamp(cosine_sim, -0.99, 0.99)

        # SUPER AGGRESSIVE: Adaptive gamma with diversity boost
        if self.training:
            sim_std = torch.std(cosine_sim)
            if sim_std < 0.01:  # Very low diversity
                adaptive_gamma = self.gamma * 3.0  # Much larger gamma
            else:
                adaptive_gamma = self.gamma * (1.0 + sim_std.detach())
        else:
            adaptive_gamma = self.gamma

        # Enhanced margin-based variance
        margin_values = torch.clamp(adaptive_gamma - cosine_sim, min=0.0, max=10.0)
        var_component = margin_values.mean(dim=-1, keepdim=True)
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))

        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # Your complex attention with SUPER AGGRESSIVE enhancements
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_sim = torch.clamp(cosine_sim, -0.99, 0.99)

            # Your complex formulas with enhanced stability
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # SUPER AGGRESSIVE: Enhanced dynamic weighting
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)

                # Temperature-controlled prediction with wider range
                weight_temp = torch.clamp(self.weight_temperature, 0.1, 20.0)
                weight_logits = self.weight_predictor(qk_features)
                weights = F.softmax(weight_logits / weight_temp, dim=-1)

                # SUPER AGGRESSIVE: Prevent extreme weight dominance
                min_weight = 0.05  # Lower minimum for more diversity
                max_weight = 0.8   # Prevent total dominance
                weights = torch.clamp(weights, min_weight, max_weight)
                weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize

                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, 0.1, 0.9)

            # SUPER AGGRESSIVE: Component combination with adaptive scaling
            cosine_norm = torch.std(cosine_sim).detach() + 1e-6
            cov_norm = torch.std(cov_component).detach() + 1e-6  
            var_norm = torch.std(var_component).detach() + 1e-6

            # Adaptive scaling based on component health
            if cosine_norm < 1e-4:  # Cosine component dead
                cosine_scale = 0.1
            else:
                cosine_scale = 1.0

            cosine_scaled = cosine_sim / cosine_norm * cosine_scale
            cov_scaled = cov_component / cov_norm * 0.5  # Stronger covariance
            var_scaled = var_component / var_norm * 0.5   # Stronger variance

            dots = (cos_weight * cosine_scaled +
                   cov_weight * cov_scaled +
                   var_weight * var_scaled)

            # SUPER AGGRESSIVE: Adaptive temperature with emergency handling
            dot_std = torch.std(dots)
            if dot_std < 1e-6:  # Attention collapsed
                attention_temperature = 0.1  # Very sensitive
            else:
                attention_temperature = 0.3 + dot_std.detach()

            dots = dots / torch.clamp(attention_temperature, 0.1, 5.0)

            out = torch.matmul(self.sm(dots), f_v)
        else:
            # Enhanced standard attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Enhanced statistics with health metrics"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:
            # Health metrics
            weight_diversity = np.std(weights, axis=0).sum()
            weight_balance = 1.0 - np.std(weights.mean(axis=0)) / 0.577  # Normalized balance

            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'diversity_score': float(weight_diversity),
                'balance_score': float(weight_balance),
                'health_status': 'healthy' if weight_diversity > 0.1 else 'stuck',
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=10, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=10, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=10, range=(0,1))[0].tolist()
                }
            }
        return {}

    def clear_weight_history(self):
        self.weight_history = []

def cosine_distance(x1, x2):
    """Enhanced cosine distance with SUPER AGGRESSIVE stability"""
    dots = torch.matmul(x1, x2)
    eps = 1e-8
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps
    scale = torch.matmul(norm1, norm2)
    result = torch.clamp(dots / scale, -0.99, 0.99)
    return result

print("✅ SUPER AGGRESSIVE TRANSFORMER WITH TENSOR FIX READY!")
