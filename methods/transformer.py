
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

class ProgressiveTracker:
    """Progressive learning tracker - replaces aggressive intervention system"""
    def __init__(self):
        self.variance_history = []
        self.learning_phase = "minimal"  # minimal, gradual, stable
        self.phase_counter = 0
        self.last_variance = 0.0
        self.learning_momentum = 0.0

    def update_learning_state(self, current_variance, epoch):
        """Gentle learning state updates instead of emergency interventions"""
        self.variance_history.append(current_variance)
        if len(self.variance_history) > 20:
            self.variance_history.pop(0)

        self.last_variance = current_variance

        # Determine learning phase based on epoch and performance
        if epoch < 15:
            self.learning_phase = "minimal"
        elif epoch < 40:
            self.learning_phase = "gradual" 
        else:
            self.learning_phase = "stable"

        # Calculate learning momentum
        if len(self.variance_history) >= 5:
            recent_mean = np.mean(self.variance_history[-5:])
            older_mean = np.mean(self.variance_history[-10:-5]) if len(self.variance_history) >= 10 else recent_mean
            self.learning_momentum = recent_mean - older_mean

        return {
            'phase': self.learning_phase,
            'momentum': self.learning_momentum,
            'stability': np.std(self.variance_history[-10:]) if len(self.variance_history) >= 10 else 0.0
        }

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

        # PROGRESSIVE: Gentle learning tracker replaces aggressive intervention
        self.learning_tracker = ProgressiveTracker()
        self.training_epoch = 0
        self.batch_count = 0
        self.learning_progress = 0.0
        self.last_variance = 0.0
        self.stable_learning = False

        # PROGRESSIVE: Start in learning mode, not emergency mode
        self.learning_mode = "gentle"  # gentle, adaptive, focused

        # PROGRESSIVE: Enhanced attention with moderate parameters
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                            initial_cov_weight=initial_cov_weight,
                            initial_var_weight=initial_var_weight,
                            dynamic_weight=dynamic_weight,
                            gamma=gamma,
                            lambda_reg=lambda_reg)

        # PROGRESSIVE: Moderate temperature system - starts normal, not extreme
        self.sm = nn.Softmax(dim=-2)
        self.temperature_sm = nn.Parameter(torch.ones(1) * 1.0)  # Normal initial temperature

        # PROGRESSIVE: Moderate initialization for steady learning
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.1)  # Moderate variance

        # PROGRESSIVE: Progressive FFN with adjustable dropout (controlled by scheduler)
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.02),  # Will be adjusted by progressive scheduler
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.02)   # Will be adjusted by progressive scheduler
        )

        # PROGRESSIVE: Progressive classification layers with adjustable dropout
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.02),  # Will be adjusted by progressive scheduler
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.02),  # Will be adjusted by progressive scheduler
                nn.Linear(dim_head, 1)
            )

    def gentle_parameter_adjustment(self, learning_state):
        """Gentle parameter adjustments based on learning state"""
        phase = learning_state['phase']
        momentum = learning_state['momentum']

        if phase == "minimal":
            # Minimal phase: very gentle adjustments
            if momentum < -0.001:  # Learning slowing down
                with torch.no_grad():
                    # Gentle temperature adjustment
                    current_temp = self.temperature_sm.item()
                    if current_temp > 2.0:
                        self.temperature_sm.data *= 0.95

                    # Gentle proto weight adjustment
                    noise_scale = 0.01
                    noise = torch.randn_like(self.proto_weight) * noise_scale
                    self.proto_weight.data += noise

        elif phase == "gradual":
            # Gradual phase: moderate adjustments
            if momentum < -0.002 and self.last_variance < 0.005:
                with torch.no_grad():
                    # Moderate temperature adjustment
                    self.temperature_sm.data *= 0.9

                    # Moderate proto weight adjustment
                    noise_scale = 0.02
                    noise = torch.randn_like(self.proto_weight) * noise_scale
                    self.proto_weight.data += noise

        # No aggressive resets - just gentle nudges
        return True

    def progressive_learning_boost(self, learning_state):
        """Progressive learning boost based on current state"""
        phase = learning_state['phase']
        stability = learning_state['stability']

        if phase == "minimal" and stability < 0.001:
            # Very gentle boost in minimal phase
            print(f"🌱 GENTLE LEARNING BOOST: Phase={phase}, Stability={stability:.6f}")

            with torch.no_grad():
                # Gentle attention boost
                if hasattr(self.ATTN, 'gentle_diversity_boost'):
                    self.ATTN.gentle_diversity_boost()

                # Small temperature adjustment
                self.temperature_sm.data = torch.clamp(
                    self.temperature_sm.data * 1.1, 0.5, 3.0)

            return True

        elif phase == "gradual" and stability < 0.002:
            # Moderate boost in gradual phase
            print(f"🌿 MODERATE LEARNING BOOST: Phase={phase}, Stability={stability:.6f}")

            with torch.no_grad():
                # Moderate attention adjustments
                if hasattr(self.ATTN, 'moderate_diversity_boost'):
                    self.ATTN.moderate_diversity_boost()

                # Moderate temperature adjustment
                self.temperature_sm.data = torch.clamp(
                    self.temperature_sm.data * 1.05, 0.3, 2.0)

            return True

        return False

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # PROGRESSIVE: Adaptive normalization based on learning phase
        learning_state = self.learning_tracker.learning_phase

        if learning_state == "minimal":
            norm_strength = 0.2  # Very light normalization
        elif learning_state == "gradual":
            norm_strength = 0.4  # Moderate normalization
        else:
            norm_strength = 0.6  # Normal normalization

        z_support = F.normalize(z_support, p=2, dim=-1) * norm_strength + z_support * (1 - norm_strength)
        z_query = F.normalize(z_query, p=2, dim=-1) * norm_strength + z_query * (1 - norm_strength)

        # PROGRESSIVE: Moderate temperature control
        if self.training:
            temp_min, temp_max = 0.3, 3.0  # Much more moderate range
        else:
            temp_min, temp_max = 0.5, 2.0

        current_temp = torch.clamp(self.temperature_sm, min=temp_min, max=temp_max)

        # Prototype computation with progressive diversity management
        proto_logits = self.proto_weight / current_temp
        proto_weights = self.sm(proto_logits)

        # PROGRESSIVE: Adaptive uniform weighting based on learning progress
        if learning_state == "minimal":
            uniform_weight = 0.3  # Higher uniform weighting for exploration
        elif learning_state == "gradual":
            uniform_weight = 0.2  # Moderate uniform weighting
        else:
            uniform_weight = 0.1  # Lower uniform weighting for exploitation

        proto_weights = proto_weights * (1 - uniform_weight) + uniform_weight / self.k_shot
        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)

        x, query = z_proto, z_query

        # PROGRESSIVE: Attention processing with adaptive scaling
        for layer_idx in range(self.depth):
            x_input = x.clone()
            attn_out = self.ATTN(q=x, k=query, v=query)

            # PROGRESSIVE: Gentle residual scaling based on learning state
            if self.training:
                if learning_state == "minimal":
                    scale_factor = 0.2  # Gentle scaling
                elif learning_state == "gradual":
                    scale_factor = 0.3  # Moderate scaling
                else:
                    scale_factor = 0.4  # Normal scaling
            else:
                scale_factor = 0.3

            x = x_input + scale_factor * attn_out

            # PROGRESSIVE: Gentle FFN processing
            ffn_out = self.FFN(x.view(-1, x.size(-1))).view(x.shape)

            if learning_state == "minimal":
                ffn_scale = 0.2  # Very gentle
            elif learning_state == "gradual":
                ffn_scale = 0.3  # Moderate
            else:
                ffn_scale = 0.4  # Normal

            x = x + ffn_scale * ffn_out

        # PROGRESSIVE: Gentle normalization
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=0.01, max=10.0)  # Much gentler clamping

        scores = self.linear(x).squeeze()

        # PROGRESSIVE: Gentle score diversity enhancement
        if self.training:
            score_std = torch.std(scores)

            # Only gentle nudges when really needed
            if score_std < 0.05 and learning_state != "minimal":
                noise_scale = min(0.1, 0.02 / (score_std + 1e-6))
                noise = torch.randn_like(scores) * noise_scale
                scores = scores + noise

                if score_std < 0.02:
                    print(f"🌱 Gentle score diversity boost: std={score_std:.6f}")

        # PROGRESSIVE: Moderate final temperature scaling
        if self.training:
            score_range = scores.max() - scores.min()
            if score_range < 0.1:
                final_temp = torch.tensor(0.5, device=scores.device, dtype=scores.dtype)
            elif score_range > 5.0:
                final_temp = torch.tensor(1.5, device=scores.device, dtype=scores.dtype)
            else:
                final_temp = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        else:
            final_temp = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

        scores = scores / final_temp
        return scores

    def set_forward_loss(self, x):
        """PROGRESSIVE: Gentle loss computation with adaptive regularization"""
        self.batch_count += 1
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)
        current_variance = torch.var(scores).item()
        self.last_variance = current_variance

        # PROGRESSIVE: Gentle learning state updates
        learning_state = self.learning_tracker.update_learning_state(
            current_variance, self.training_epoch)

        # PROGRESSIVE: Gentle parameter adjustments instead of nuclear resets
        if learning_state['momentum'] < -0.003:  # Only if really struggling
            adjustment_applied = self.gentle_parameter_adjustment(learning_state)

            if adjustment_applied:
                # Re-compute scores after gentle adjustment
                scores = self.set_forward(x)
                new_variance = torch.var(scores).item()
                self.last_variance = new_variance

        # PROGRESSIVE: Gentle learning boosts
        if current_variance < 0.002 and learning_state['stability'] < 0.001:
            boost_applied = self.progressive_learning_boost(learning_state)
            if boost_applied:
                scores = self.set_forward(x)
                current_variance = torch.var(scores).item()
                self.last_variance = current_variance

        # PROGRESSIVE: Base loss
        base_loss = self.loss_fn(scores, target)

        # PROGRESSIVE: Progressive regularization based on learning phase
        if learning_state['phase'] == "minimal":
            # Minimal phase: pure learning, no regularization
            total_loss = base_loss

        elif learning_state['phase'] == "gradual":
            # Gradual phase: gentle regularization
            if current_variance < 0.003:
                probs = F.softmax(scores, dim=1)
                score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                entropy_bonus = 0.01 * score_entropy  # Very gentle
                total_loss = base_loss - entropy_bonus
            else:
                total_loss = base_loss

        else:
            # Stable phase: moderate regularization
            if current_variance < 0.005:
                probs = F.softmax(scores, dim=1)
                score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                entropy_bonus = 0.02 * score_entropy  # Moderate
                total_loss = base_loss - entropy_bonus
            else:
                total_loss = base_loss

        # PROGRESSIVE: Update learning mode
        if current_variance > 0.01 and self.training_epoch > 5:
            if not self.stable_learning:
                print(f"🌟 STABLE LEARNING ACHIEVED - Variance: {current_variance:.6f}")
                self.stable_learning = True

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)

        return acc, total_loss

class Attention(nn.Module):
    """PROGRESSIVE: Moderate attention with gentle learning enhancements"""
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

        # PROGRESSIVE: Moderate dynamic weighting system
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head * 2),  # Moderate size
                nn.LayerNorm(dim_head * 2),
                nn.ReLU(),
                nn.Dropout(0.02),  # Will be controlled by progressive scheduler
                nn.Linear(dim_head * 2, dim_head),
                nn.ReLU(),
                nn.Dropout(0.02),
                nn.Linear(dim_head, 3),
                nn.Softmax(dim=-1)
            )

            self.weight_temperature = nn.Parameter(torch.ones(1) * 0.5)  # Moderate sensitivity
            self.weight_history = []
            # PROGRESSIVE: Start with balanced weights
            self.last_weights = torch.tensor([0.4, 0.3, 0.3])  # More balanced
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # PROGRESSIVE: Moderate processing with controllable dropout
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.02)  # Will be controlled by progressive scheduler
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.02)  # Will be controlled by progressive scheduler
        ) if project_out else nn.Identity()

        self.record_weights = False

    def gentle_diversity_boost(self):
        """PROGRESSIVE: Gentle diversity boost for minimal phase"""
        if self.dynamic_weight:
            print("🌱 GENTLE: Soft parameter adjustment")
            with torch.no_grad():
                # Gentle parameter adjustments
                for param in self.weight_predictor.parameters():
                    noise_scale = 0.02  # Very gentle
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)

                # Gentle temperature adjustment
                self.weight_temperature.data = torch.clamp(
                    self.weight_temperature.data * 1.1, 0.2, 1.0)

    def moderate_diversity_boost(self):
        """PROGRESSIVE: Moderate diversity boost for gradual phase"""
        if self.dynamic_weight:
            print("🌿 MODERATE: Medium parameter adjustment")
            with torch.no_grad():
                # Moderate parameter adjustments
                for param in self.weight_predictor.parameters():
                    noise_scale = min(0.05, param.std().item() * 0.1)
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)

                # Moderate temperature adjustment
                self.weight_temperature.data = torch.clamp(
                    self.weight_temperature.data * 1.05, 0.1, 0.8)

    def force_balanced_learning(self):
        """PROGRESSIVE: Encourage balanced component usage"""
        if self.dynamic_weight:
            print("🎯 BALANCE: Encouraging balanced component usage")
            with torch.no_grad():
                # Get final layer parameters
                final_layers = [m for m in self.weight_predictor.modules() if isinstance(m, nn.Linear)]
                if final_layers:
                    final_layer = final_layers[-1]
                    if final_layer.out_features == 3:
                        # Encourage more balanced weights
                        balanced_weight = torch.tensor([
                            [1.0, 0.5, 0.5],   # Moderate cosine preference
                            [0.5, 1.0, 0.5],   # Moderate covariance preference  
                            [0.5, 0.5, 1.0]    # Moderate variance preference
                        ], device=final_layer.weight.device)

                        # Gentle adjustment toward balanced weights
                        final_layer.weight.data = final_layer.weight.data * 0.8 + balanced_weight * 0.2

                        if final_layer.bias is not None:
                            balanced_bias = torch.tensor([0.0, 0.0, 0.0], device=final_layer.bias.device)
                            final_layer.bias.data = final_layer.bias.data * 0.8 + balanced_bias * 0.2

    def get_current_weight_diversity(self):
        """Get current diversity of dynamic weights"""
        if self.dynamic_weight and len(self.weight_history) > 0:
            recent_weights = self.weight_history[-10:] if len(self.weight_history) >= 10 else self.weight_history
            if recent_weights:
                weights_array = np.array(recent_weights)
                return np.std(weights_array)
        return 0.0

    def compute_regularized_covariance(self, f_q, f_k):
        """PROGRESSIVE: Moderate covariance computation"""
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # PROGRESSIVE: Gentle centering
        f_k_centered = f_k - f_k.mean(dim=2, keepdim=True) 
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)

        # PROGRESSIVE: Moderate covariance computation
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5 + 1e-6)

        # PROGRESSIVE: Gentle regularization
        regularization_factor = self.lambda_reg / max(m, 1)
        cov_component = regularization_factor * cov_component
        cov_component = torch.clamp(cov_component, -10.0, 10.0)  # Gentler clamping

        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """PROGRESSIVE: Moderate variance computation"""
        # PROGRESSIVE: Gentle normalization
        f_q_norm = F.normalize(f_q, p=2, dim=-1, eps=1e-6)
        f_k_norm = F.normalize(f_k, p=2, dim=-1, eps=1e-6)

        # PROGRESSIVE: Moderate cosine similarity
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))
        cosine_sim = torch.clamp(cosine_sim, -0.9, 0.9)  # Safer range

        # PROGRESSIVE: Moderate adaptive gamma
        if self.training:
            sim_std = torch.std(cosine_sim).item()
            if sim_std < 0.005:  # More reasonable threshold
                adaptive_gamma = self.gamma * 5.0  # Moderate boost
            elif sim_std < 0.02:
                adaptive_gamma = self.gamma * 3.0  # Moderate boost
            else:
                adaptive_gamma = self.gamma * (1.0 + sim_std * 5.0)  # Gentler scaling
        else:
            adaptive_gamma = self.gamma

        # PROGRESSIVE: Moderate margin-based variance
        margin_values = torch.clamp(adaptive_gamma - cosine_sim, min=0.0, max=5.0)  # Gentler max
        var_component = margin_values.mean(dim=-1, keepdim=True)
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))

        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # PROGRESSIVE: Moderate attention computation
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_sim = torch.clamp(cosine_sim, -0.9, 0.9)

            # Compute components with moderate settings
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # PROGRESSIVE: Moderate dynamic weighting
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)

                # PROGRESSIVE: Moderate temperature control
                weight_temp = torch.clamp(self.weight_temperature, 0.1, 2.0)
                weight_logits = self.weight_predictor(qk_features)
                weights = F.softmax(weight_logits / weight_temp, dim=-1)

                # PROGRESSIVE: Moderate weight constraints
                min_weight = 0.05  # Reasonable minimum
                max_weight = 0.8   # Reasonable maximum
                weights = torch.clamp(weights, min_weight, max_weight)
                weights = weights / weights.sum(dim=-1, keepdim=True)

                # PROGRESSIVE: Track weight evolution
                if self.training:
                    current_weights = weights.detach().cpu().numpy().mean(axis=0)
                    self.weight_history.append(current_weights)
                    if len(self.weight_history) > 100:  # Reasonable history
                        self.weight_history.pop(0)
                    self.last_weights = torch.tensor(current_weights)

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, 0.1, 0.8)

            # PROGRESSIVE: Moderate component combination
            cosine_norm = torch.std(cosine_sim).detach() + 1e-6
            cov_norm = torch.std(cov_component).detach() + 1e-6
            var_norm = torch.std(var_component).detach() + 1e-6

            # PROGRESSIVE: Moderate scaling
            cosine_scaled = cosine_sim / cosine_norm
            cov_scaled = cov_component / cov_norm * 0.5  # Moderate scaling
            var_scaled = var_component / var_norm * 0.5   # Moderate scaling

            # Combine all components
            dots = (cos_weight * cosine_scaled + 
                   cov_weight * cov_scaled + 
                   var_weight * var_scaled)

            # PROGRESSIVE: Moderate attention temperature
            dots_std = torch.std(dots)
            if dots_std < 1e-4:  # Very low
                attention_temperature = torch.tensor(0.1, device=dots.device, dtype=dots.dtype)
            elif dots_std < 1e-2:  # Low
                attention_temperature = torch.tensor(0.3, device=dots.device, dtype=dots.dtype)
            else:  # Normal
                attention_temperature = torch.tensor(0.5, device=dots.device, dtype=dots.dtype) + dots_std

            attention_temperature = torch.clamp(attention_temperature, 0.1, 3.0)
            dots = dots / attention_temperature

            out = torch.matmul(self.sm(dots), f_v)

        else:
            # PROGRESSIVE: Standard attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Enhanced statistics for progressive learning"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:
            # PROGRESSIVE: Moderate metrics
            weight_diversity = np.std(weights, axis=0).sum()
            weight_balance = 1.0 - np.std(weights.mean(axis=0)) / 0.577

            learning_trend = 0.0
            if len(weights) >= 10:
                recent_diversity = np.std(weights[-10:], axis=0).sum()
                old_diversity = np.std(weights[:-10], axis=0).sum() if len(weights) > 10 else 0
                learning_trend = recent_diversity - old_diversity

            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'diversity_score': float(weight_diversity),
                'balance_score': float(weight_balance),
                'learning_trend': float(learning_trend),
                'health_status': 'stable' if weight_diversity > 0.03 else 'learning' if weight_diversity > 0.01 else 'gentle_nudge_needed',
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=5, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=5, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=5, range=(0,1))[0].tolist()
                }
            }
        return {}

    def clear_weight_history(self):
        """PROGRESSIVE: Gentle history management"""
        if len(self.weight_history) > 20:
            self.weight_history = self.weight_history[-10:]  # Keep reasonable history
        else:
            self.weight_history = []

def cosine_distance(x1, x2):
    """PROGRESSIVE: Stable cosine distance computation"""
    dots = torch.matmul(x1, x2)
    eps = 1e-6  # Reasonable epsilon
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps
    scale = torch.matmul(norm1, norm2)
    result = torch.clamp(dots / scale, -0.9, 0.9)  # Safe numerical range
    return result
