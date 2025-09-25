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

        # DIVERSITY PRESERVATION: Track training state for interventions
        self.training_epoch = 0
        self.last_variance = 0.0
        self.variance_history = []
        self.intervention_count = 0

        # Keep all your innovations with emergency diversity preservation
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight,
                             gamma=gamma,
                             lambda_reg=lambda_reg)

        # Softer softmax with emergency temperature
        self.sm = nn.Softmax(dim=-2)
        self.temperature_sm = nn.Parameter(torch.ones(1) * 5.0)  # Lower initial temp

        # Better prototype initialization with diversity enforcement
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.1)  # Larger init

        # Enhanced FFN with emergency diversity features
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # More dropout for regularization
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.2)
        )

        # Enhanced classification with diversity pressure
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.1),
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_head, 1)
            )

    def emergency_diversity_intervention(self, current_variance):
        """EMERGENCY: Intervene when variance collapses"""
        target_variance = 0.005  # Lower threshold for intervention

        if current_variance < target_variance * 0.1:  # Critical collapse
            self.intervention_count += 1
            print(f"🚨 CRITICAL DIVERSITY COLLAPSE! Intervention #{self.intervention_count}")

            # INTERVENTION 1: Inject noise into prototype weights
            with torch.no_grad():
                noise_scale = 0.1
                noise = torch.randn_like(self.proto_weight) * noise_scale
                self.proto_weight.data.add_(noise)
                print(f"💥 Injected {noise_scale} noise into prototype weights")

            # INTERVENTION 2: Reset temperature to encourage exploration
            with torch.no_grad():
                self.temperature_sm.data.fill_(10.0)  # Reset to higher temperature
                print(f"🌡️  Reset temperature to 10.0")

            # INTERVENTION 3: Force dynamic weight diversity
            self.ATTN.emergency_unfreeze_weights()

            return True

        elif current_variance < target_variance * 0.5:  # Moderate collapse
            print(f"⚠️  Moderate variance drop detected: {current_variance:.6f}")

            # Gentle intervention: Increase temperature
            with torch.no_grad():
                self.temperature_sm.data.clamp_(min=2.0)
                print(f"🌡️  Temperature boosted to {self.temperature_sm.item():.2f}")

            return False

        return False

    def update_training_state(self, epoch, current_variance):
        """Update training state for adaptive interventions"""
        self.training_epoch = epoch
        self.last_variance = current_variance
        self.variance_history.append(current_variance)

        # Keep only last 10 epochs of history
        if len(self.variance_history) > 10:
            self.variance_history.pop(0)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # DIVERSITY PRESERVATION: Lighter normalization
        norm_strength = 0.8  # Reduced from 0.9
        z_support = F.normalize(z_support, p=2, dim=-1) * norm_strength + z_support * (1 - norm_strength)
        z_query = F.normalize(z_query, p=2, dim=-1) * norm_strength + z_query * (1 - norm_strength)

        # EMERGENCY: Dynamic temperature control
        current_temp = torch.clamp(self.temperature_sm, min=0.5, max=50.0)

        # Prototype computation with diversity enforcement
        proto_logits = self.proto_weight / current_temp
        proto_weights = self.sm(proto_logits)

        # DIVERSITY BOOST: Add small amount of uniform weighting
        uniform_weight = 0.1
        proto_weights = proto_weights * (1 - uniform_weight) + uniform_weight / self.k_shot

        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)

        x, query = z_proto, z_query

        # Enhanced attention with diversity preservation
        for layer_idx in range(self.depth):
            x_input = x.clone()

            # Apply attention with emergency diversity features
            attn_out = self.ATTN(q=x, k=query, v=query)

            # DIVERSITY PRESERVATION: Adaptive residual scaling
            if self.training:
                # Encourage larger residuals when variance is low
                variance_boost = max(1.0, 5.0 - self.last_variance * 1000)
                scale_factor = 0.1 * variance_boost
                scale_factor = min(scale_factor, 0.5)  # Cap at 0.5
            else:
                scale_factor = 0.1

            x = x_input + scale_factor * attn_out

            # Enhanced FFN
            ffn_out = self.FFN(x.view(-1, x.size(-1))).view(x.shape)
            x = x + 0.1 * ffn_out

        # Final processing with diversity enforcement
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=0.05, max=20.0)  # More permissive normalization

        scores = self.linear(x).squeeze()  # (q, n)

        # EMERGENCY: Prevent score collapse
        if self.training:
            score_std = torch.std(scores)
            if score_std < 0.01:  # Emergency diversity injection
                noise_scale = max(0.1, 0.1 / (score_std + 1e-8))
                noise = torch.randn_like(scores) * noise_scale
                scores = scores + noise
                print(f"🆘 EMERGENCY SCORE DIVERSITY: Added {noise_scale:.3f} noise")

        # Adaptive temperature scaling based on score distribution
        score_range = scores.max() - scores.min()
        if score_range < 0.1:  # Scores too compressed
            temperature = 0.1  # Make more sensitive
        else:
            temperature = 1.0 + 0.3 * torch.tanh(score_range)

        scores = scores / temperature

        return scores

    def set_forward_loss(self, x):
        """Enhanced loss with diversity preservation"""
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)

        # Monitor variance for interventions
        current_variance = torch.var(scores).item()

        # Emergency intervention if needed
        intervention_triggered = self.emergency_diversity_intervention(current_variance)

        # Re-compute scores if intervention was triggered
        if intervention_triggered:
            scores = self.set_forward(x)
            current_variance = torch.var(scores).item()
            print(f"📊 Post-intervention variance: {current_variance:.6f}")

        # Base loss with enhanced label smoothing
        base_loss = self.loss_fn(scores, target)

        # DIVERSITY-PRESERVING LABEL SMOOTHING
        if self.training:
            # Adaptive smoothing based on variance
            variance_factor = max(0.05, min(0.25, 0.1 + current_variance * 10))

            # Create smoothed targets
            num_classes = scores.size(1)
            smooth_target = torch.full_like(scores, variance_factor / num_classes)

            # Add the main target weight
            batch_indices = torch.arange(target.size(0), device=target.device)
            smooth_target[batch_indices, target] += (1.0 - variance_factor)

            # Compute smooth loss
            log_probs = F.log_softmax(scores, dim=1)
            smooth_loss = -torch.sum(log_probs * smooth_target, dim=1).mean()

            total_loss = 0.7 * base_loss + 0.3 * smooth_loss
        else:
            total_loss = base_loss

        # ENHANCED DIVERSITY REGULARIZATION
        if self.training:
            probs = F.softmax(scores, dim=1)

            # 1. Encourage uniform class distribution
            class_probs = probs.mean(dim=0)
            uniform_target = torch.ones_like(class_probs) / self.n_way
            diversity_loss = F.kl_div(torch.log(class_probs + 1e-8), uniform_target, reduction='sum')

            # 2. Encourage score variance (entropy bonus)
            score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy_bonus = 0.1 * score_entropy

            # 3. Variance preservation loss
            variance_loss = -torch.log(torch.var(scores) + 1e-8)  # Use tensor directly  # Negative log variance

            total_loss = total_loss + 0.01 * diversity_loss + entropy_bonus + 0.001 * variance_loss

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)

        return acc, total_loss

class Attention(nn.Module):
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

        # Enhanced dynamic weighting with emergency features
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Dropout(0.3),  # More aggressive dropout
                nn.Linear(dim_head, dim_head // 2),
                nn.ReLU(),
                nn.Linear(dim_head // 2, 3),
                nn.Softmax(dim=-1)
            )
            # Dynamic temperature for weights
            self.weight_temperature = nn.Parameter(torch.ones(1) * 2.0)  # Lower initial temp
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # Enhanced input processing
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.15)
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.15)
        ) if project_out else nn.Identity()

        self.weight_history = []
        self.record_weights = False

    def emergency_unfreeze_weights(self):
        """EMERGENCY: Force dynamic weights to start learning"""
        if self.dynamic_weight:
            with torch.no_grad():
                # Inject noise into weight predictor to break symmetry
                for param in self.weight_predictor.parameters():
                    if param.dim() > 1:
                        noise = torch.randn_like(param) * 0.05  # More aggressive noise
                        param.add_(noise)

                # Reset weight temperature
                self.weight_temperature.data.fill_(1.0)

                print(f"🎛️  EMERGENCY: Dynamic weights unfrozen with noise injection")

    def compute_regularized_covariance(self, f_q, f_k):
        """Your covariance formula with enhanced stability"""
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # Enhanced numerical stability
        E_bar = f_k.mean(dim=2, keepdim=True)
        f_k_centered = f_k - E_bar
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)

        # Compute covariance with better regularization
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5 + 1e-6)

        # Adaptive regularization
        regularization_factor = self.lambda_reg / max(m, 1)
        cov_component = regularization_factor * cov_component

        # More permissive clamping for diversity
        cov_component = torch.clamp(cov_component, -20.0, 20.0)

        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """Your variance formula with enhanced diversity preservation"""
        # Lighter normalization for diversity
        f_q_norm = F.normalize(f_q, p=2, dim=-1, eps=1e-6)
        f_k_norm = F.normalize(f_k, p=2, dim=-1, eps=1e-6)

        # Cosine similarity with better range
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))
        cosine_sim = torch.clamp(cosine_sim, -0.95, 0.95)  # More permissive

        # Adaptive gamma for diversity preservation
        if self.training:
            # Smaller gamma when we need more diversity
            adaptive_gamma = self.gamma * (0.5 + 0.5 * torch.std(cosine_sim).detach())
        else:
            adaptive_gamma = self.gamma

        # Margin-based variance with diversity boost
        margin_values = torch.clamp(adaptive_gamma - cosine_sim, min=0.0, max=5.0)  # Higher max

        # Enhanced variance computation
        var_component = margin_values.mean(dim=-1, keepdim=True)
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))

        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # Enhanced cosine attention with your complex formulas
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_sim = torch.clamp(cosine_sim, -0.95, 0.95)

            # Your complex formulas with enhanced stability
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # Enhanced dynamic weighting
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)

                # Temperature-controlled weight prediction with diversity boost
                weight_temp = torch.clamp(self.weight_temperature, 0.1, 10.0)
                weight_logits = self.weight_predictor(qk_features)
                weights = F.softmax(weight_logits / weight_temp, dim=-1)

                # DIVERSITY BOOST: Prevent weights from being too extreme
                min_weight = 0.1
                weights = weights * (1 - 3 * min_weight) + min_weight

                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, 0.2, 0.8)

            # Enhanced component combination with diversity preservation
            cosine_norm = torch.std(cosine_sim).detach() + 1e-6
            cov_norm = torch.std(cov_component).detach() + 1e-6  
            var_norm = torch.std(var_component).detach() + 1e-6

            # More balanced scaling for diversity
            cosine_scaled = cosine_sim / cosine_norm
            cov_scaled = cov_component / cov_norm * 0.3  # Increased from 0.1
            var_scaled = var_component / var_norm * 0.3  # Increased from 0.1

            # Combine components
            dots = (cos_weight * cosine_scaled +
                   cov_weight * cov_scaled +
                   var_weight * var_scaled)

            # Adaptive temperature for diversity
            attention_temperature = 0.5 + torch.std(dots).detach()  # More sensitive
            dots = dots / torch.clamp(attention_temperature, 0.3, 3.0)

            out = torch.matmul(self.sm(dots), f_v)
        else:
            # Enhanced standard attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Enhanced weight statistics with diversity metrics"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:
            # Calculate diversity metrics
            weight_diversity = np.std(weights, axis=0).sum()  # Total variation across time

            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'diversity_score': float(weight_diversity),  # New metric
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
    """Enhanced cosine distance with diversity preservation"""
    dots = torch.matmul(x1, x2)
    eps = 1e-6  # Smaller epsilon for more sensitivity
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps
    scale = torch.matmul(norm1, norm2)
    result = torch.clamp(dots / scale, -0.95, 0.95)  # More permissive range
    return result

print("🚀 ULTIMATE DIVERSITY-PRESERVING TRANSFORMER CREATED!")
print("\n🎯 EMERGENCY INTERVENTIONS BUILT-IN:")
print("✅ 1. Automatic diversity collapse detection")
print("✅ 2. Emergency parameter noise injection")
print("✅ 3. Dynamic weight symmetry breaking")
print("✅ 4. Adaptive temperature controls")
print("✅ 5. Enhanced variance preservation")
print("✅ 6. Intelligent regularization adaptation")
print("✅ 7. Multi-layer diversity enforcement")
print("✅ 8. Real-time intervention system")

print("\n📈 DIVERSITY PRESERVATION FEATURES:")
print("🛡️  Emergency variance monitoring")
print("🎛️  Automatic dynamic weight unfreezing")
print("🌡️  Adaptive temperature scaling")
print("💥 Parameter noise injection")
print("🎯 Enhanced entropy regularization")
print("📊 Variance preservation loss")
print("⚖️  Balanced component weighting")

print("\n🚨 INTERVENTION SYSTEM:")
print("When variance drops below thresholds:")
print("• <0.0005: CRITICAL intervention (noise + reset)")
print("• <0.0025: MODERATE intervention (temperature boost)")
print("• Real-time adaptation during training")
print("• Automatic recovery mechanisms")

print("\nThis transformer will FIGHT the variance collapse!")