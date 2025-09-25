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
        self.stuck_counter = 0
        self.last_intervention_variance = 0.0

    def should_intervene(self, current_variance, batch_idx):
        """Smart intervention decision based on variance trends"""
        # Always track variance history
        self.variance_history.append(current_variance)
        if len(self.variance_history) > 30: # Keep last 30 measurements
            self.variance_history.pop(0)

        # Reduce cooldown
        if self.intervention_cooldown > 0:
            self.intervention_cooldown -= 1
            return False, "cooling_down"

        # Check if we're truly stuck (variance not improving over time)
        if len(self.variance_history) >= 15:
            recent_vars = self.variance_history[-15:]
            old_vars = self.variance_history[-30:-15] if len(self.variance_history) >= 30 else self.variance_history[:-15]
            recent_mean = np.mean(recent_vars)
            old_mean = np.mean(old_vars) if old_vars else recent_mean

            # BREAKTHROUGH FIX: More aggressive stuck detection
            if recent_mean < 0.005 and abs(recent_mean - old_mean) < 0.001:  # More sensitive
                self.stuck_counter += 1
                if self.stuck_counter >= 2:  # Faster stuck detection
                    self.stuck_counter = 0
                    self.intervention_cooldown = 15  # Shorter cooldown
                    self.last_intervention_variance = current_variance
                    return True, "truly_stuck"
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # BREAKTHROUGH FIX: More aggressive emergency intervention
        if current_variance < 0.001:  # Lower threshold
            self.intervention_cooldown = 8  # Shorter cooldown
            self.last_intervention_variance = current_variance
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

        # BREAKTHROUGH SYSTEM: Smart intervention tracking
        self.intervention_tracker = InterventionTracker()
        self.training_epoch = 0
        self.batch_count = 0
        self.major_interventions = 0
        self.last_variance = 0.0
        self.breakthrough_achieved = False

        # BREAKTHROUGH: Enhanced attention with all your innovations
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight,
                             gamma=gamma,
                             lambda_reg=lambda_reg)

        # BREAKTHROUGH: Smart temperature system
        self.sm = nn.Softmax(dim=-2)
        self.temperature_sm = nn.Parameter(torch.ones(1) * 2.5) # Start moderate

        # BREAKTHROUGH: Enhanced initialization for diversity
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.15) # Good balance

        # BREAKTHROUGH: Enhanced FFN with controlled dropout
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.20),  # Reduced dropout for better learning
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.20)   # Reduced dropout
        )

        # Enhanced classification layers
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.12),  # Reduced dropout
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.12),  # Reduced dropout
                nn.Linear(dim_head, 1)
            )

    def breakthrough_intervention(self, current_variance, intervention_type):
        """BREAKTHROUGH: Smart interventions that actually work"""
        if intervention_type == "emergency":
            self.major_interventions += 1
            print(f"\n🚨 EMERGENCY BREAKTHROUGH INTERVENTION #{self.major_interventions}")
            print(f"   Current variance: {current_variance:.8f} (CRITICAL LOW)")
            
            with torch.no_grad():
                # MASSIVE parameter shake-up
                noise_scale = 0.6  # Even larger noise
                proto_noise = torch.randn_like(self.proto_weight) * noise_scale
                self.proto_weight.data.add_(proto_noise)

                # DRAMATIC temperature reset
                self.temperature_sm.data.fill_(25.0)  # Very high exploration

                # Force dynamic weight breakthrough
                self.ATTN.nuclear_breakthrough()

                print(f"   💥 MASSIVE parameter noise: {noise_scale}")
                print(f"   🌡️ DRAMATIC temperature reset: 25.0")
                print(f"   ☢️ NUCLEAR weight breakthrough applied")
            return True

        elif intervention_type == "truly_stuck":
            self.major_interventions += 1
            print(f"\n⚡ MAJOR BREAKTHROUGH INTERVENTION #{self.major_interventions}")
            print(f"   Current variance: {current_variance:.8f} (CONFIRMED STUCK)")
            
            with torch.no_grad():
                # Progressive noise injection based on how stuck we are
                base_noise = 0.35  # Increased base noise
                stuck_multiplier = min(2.5, 1.0 + self.major_interventions * 0.3)  # More aggressive
                noise_scale = base_noise * stuck_multiplier
                proto_noise = torch.randn_like(self.proto_weight) * noise_scale
                self.proto_weight.data.add_(proto_noise)

                # Smart temperature scaling
                current_temp = self.temperature_sm.item()
                if current_temp < 20.0:
                    new_temp = min(current_temp * 2.2, 25.0)  # More aggressive scaling
                    self.temperature_sm.data.fill_(new_temp)
                    print(f"   🌡️ Temperature BOOSTED: {current_temp:.2f} → {new_temp:.2f}")
                else:
                    # If temperature already high, try parameter diversity injection
                    self.ATTN.emergency_diversity_injection()
                    print(f"   💥 Alternative: Emergency diversity injection applied")

                # Force dynamic weight learning
                self.ATTN.force_dynamic_learning()
                print(f"   🎛️ Forced dynamic weight learning")

                # Update breakthrough status
                if current_variance > 0.003:  # Lower threshold for breakthrough
                    print(f"   🎉 BREAKTHROUGH DETECTED!")
                    self.breakthrough_achieved = True

            return True

        return False

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # BREAKTHROUGH: Balanced normalization for stable diversity
        norm_strength = 0.8  # Slightly stronger normalization
        z_support = F.normalize(z_support, p=2, dim=-1) * norm_strength + z_support * (1 - norm_strength)
        z_query = F.normalize(z_query, p=2, dim=-1) * norm_strength + z_query * (1 - norm_strength)

        # Smart temperature control with wide but controlled range
        current_temp = torch.clamp(self.temperature_sm, min=0.1, max=50.0)

        # Prototype computation with diversity enforcement
        proto_logits = self.proto_weight / current_temp
        proto_weights = self.sm(proto_logits)

        # BREAKTHROUGH: Adaptive uniform weighting based on training progress
        if self.breakthrough_achieved:
            uniform_weight = 0.08  # Less uniform weighting after breakthrough
        else:
            uniform_weight = 0.22  # More uniform weighting before breakthrough

        proto_weights = proto_weights * (1 - uniform_weight) + uniform_weight / self.k_shot
        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)
        x, query = z_proto, z_query

        # Enhanced attention with breakthrough-aware processing
        for layer_idx in range(self.depth):
            x_input = x.clone()
            attn_out = self.ATTN(q=x, k=query, v=query)

            # BREAKTHROUGH: Variance-adaptive residual scaling
            if self.training:
                if self.last_variance < 0.0008:  # Very low variance
                    variance_boost = 8.0  # Even stronger boost
                elif self.last_variance < 0.003:  # Low variance
                    variance_boost = 3.5  # Stronger boost
                else:  # Healthy variance
                    variance_boost = 1.0  # Normal scaling

                scale_factor = 0.18 * variance_boost
                scale_factor = min(scale_factor, 0.5)  # Higher cap
            else:
                scale_factor = 0.18

            x = x_input + scale_factor * attn_out

            # Enhanced FFN with adaptive processing
            ffn_out = self.FFN(x.view(-1, x.size(-1))).view(x.shape)
            
            # Adaptive FFN residual based on breakthrough status
            ffn_scale = 0.28 if self.breakthrough_achieved else 0.22  # Higher scaling
            x = x + ffn_scale * ffn_out

        # BREAKTHROUGH: Smart final processing
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=0.01, max=40.0)  # Wider range

        scores = self.linear(x).squeeze()

        # BREAKTHROUGH: Emergency score diversity system
        if self.training:
            score_std = torch.std(scores)
            if score_std < 0.05:  # Scores too uniform
                noise_scale = max(0.25, 0.25 / (score_std + 1e-8))  # More aggressive
                noise_scale = min(noise_scale, 2.0)  # Higher cap
                noise = torch.randn_like(scores) * noise_scale
                scores = scores + noise

                if score_std < 0.02:  # Really critical
                    print(f"🆘 EMERGENCY SCORE DIVERSITY: std={score_std:.6f}, added_noise={noise_scale:.3f}")

        # BREAKTHROUGH: Adaptive final temperature scaling - TENSOR SAFE
        score_range = scores.max() - scores.min()
        if score_range < 0.15:  # Compressed scores
            final_temp = torch.tensor(0.08, device=scores.device, dtype=scores.dtype)  # Lower temp
        elif score_range > 10.0:  # Over-spread scores
            final_temp = torch.tensor(2.5, device=scores.device, dtype=scores.dtype)  # Higher temp
        else:
            final_temp = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

        scores = scores / final_temp
        return scores

    def set_forward_loss(self, x):
        """BREAKTHROUGH: Enhanced loss with smart diversity preservation"""
        self.batch_count += 1
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)
        current_variance = torch.var(scores).item()
        self.last_variance = current_variance

        # BREAKTHROUGH: Smart intervention system
        should_intervene, intervention_type = self.intervention_tracker.should_intervene(
            current_variance, self.batch_count)

        if should_intervene and intervention_type != "cooling_down":
            intervention_applied = self.breakthrough_intervention(current_variance, intervention_type)
            if intervention_applied:
                # Re-compute scores after intervention
                scores = self.set_forward(x)
                new_variance = torch.var(scores).item()
                print(f"   📊 Variance improvement: {current_variance:.6f} → {new_variance:.6f}")
                current_variance = new_variance
                self.last_variance = current_variance

        # Enhanced base loss
        base_loss = self.loss_fn(scores, target)

        # BREAKTHROUGH: Smart adaptive label smoothing
        if self.training:
            # More aggressive smoothing for stuck states
            if current_variance < 0.0008:
                smooth_factor = 0.4  # Very high smoothing
            elif current_variance < 0.003:
                smooth_factor = 0.3  # High smoothing
            elif current_variance < 0.008:
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

        # BREAKTHROUGH: Enhanced diversity regularization system
        if self.training:
            probs = F.softmax(scores, dim=1)

            # 1. Class distribution uniformity
            class_probs = probs.mean(dim=0)
            uniform_target = torch.ones_like(class_probs) / self.n_way
            diversity_loss = F.kl_div(torch.log(class_probs + 1e-8), uniform_target, reduction='sum')

            # 2. Enhanced entropy bonus
            score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy_bonus = 0.2 * score_entropy  # Stronger entropy encouragement

            # 3. Variance preservation with adaptive strength
            variance_tensor = torch.var(scores)
            if variance_tensor.item() < 0.0008:  # Very low variance
                variance_loss = -4.0 * torch.log(variance_tensor + 1e-8)  # Stronger penalty
            elif variance_tensor.item() < 0.003:  # Low variance
                variance_loss = -2.0 * torch.log(variance_tensor + 1e-8)  # Strong penalty
            else:  # Healthy variance
                variance_loss = -0.5 * torch.log(variance_tensor + 1e-8)  # Light penalty

            # 4. Dynamic weight learning bonus
            weight_bonus = 0.0
            if hasattr(self.ATTN, 'get_current_weight_diversity'):
                weight_diversity = self.ATTN.get_current_weight_diversity()
                if weight_diversity < 0.02:  # Weights are stuck
                    weight_bonus = -0.8 * torch.log(torch.tensor(weight_diversity + 1e-8, device=scores.device))

            # Combine all regularization terms with stronger weights
            total_loss = total_loss + 0.02 * diversity_loss + entropy_bonus + 0.002 * variance_loss + 0.002 * weight_bonus

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, total_loss

class Attention(nn.Module):
    """BREAKTHROUGH: Enhanced attention with all your innovations + breakthrough systems"""
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

        # BREAKTHROUGH: Enhanced dynamic weighting system with more powerful architecture
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head * 3),  # Larger first layer
                nn.LayerNorm(dim_head * 3),
                nn.ReLU(),
                nn.Dropout(0.15),  # Lower dropout for better learning
                nn.Linear(dim_head * 3, dim_head * 2),
                nn.ReLU(), 
                nn.Dropout(0.1),   # Even lower dropout
                nn.Linear(dim_head * 2, 3),
                nn.Softmax(dim=-1)
            )
            self.weight_temperature = nn.Parameter(torch.ones(1) * 1.8)  # Higher initial temp
            self.weight_history = []
            # BREAKTHROUGH: Start with asymmetric weights to break symmetry
            self.last_weights = torch.tensor([0.4, 0.25, 0.35])
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # Enhanced processing layers with lower dropout
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.12)  # Reduced dropout
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.12)  # Reduced dropout
        ) if project_out else nn.Identity()

        self.record_weights = False

    def nuclear_breakthrough(self):
        """NUCLEAR OPTION: Complete system reset for breakthrough"""
        if self.dynamic_weight:
            with torch.no_grad():
                # Complete reinitialization with controlled randomness
                for param in self.weight_predictor.parameters():
                    if param.dim() > 1:
                        # Xavier initialization with extra variance for diversity
                        nn.init.xavier_uniform_(param, gain=1.5)  # Higher gain
                    else:
                        # Asymmetric bias initialization
                        param.uniform_(-0.05, 0.05)

                # Reset temperature for more exploration
                self.weight_temperature.data.fill_(0.2)  # Lower temperature = more sensitivity

                # Clear history
                self.weight_history = []
                print(f"☢️ NUCLEAR: Complete weight predictor breakthrough reset")

    def emergency_diversity_injection(self):
        """Emergency parameter diversity injection"""
        if self.dynamic_weight:
            with torch.no_grad():
                # Inject controlled noise into all predictor parameters
                for param in self.weight_predictor.parameters():
                    param_std = param.std().item()
                    noise_scale = max(0.08, min(0.25, param_std * 0.5))  # More aggressive noise
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)

                # Adjust temperature for better sensitivity
                current_temp = self.weight_temperature.item()
                new_temp = max(0.1, current_temp * 0.7)  # Make more sensitive
                self.weight_temperature.data.fill_(new_temp)
                print(f"💥 EMERGENCY: Diversity injection applied, temp: {current_temp:.3f} → {new_temp:.3f}")

    def force_dynamic_learning(self):
        """Force dynamic weights to start learning by breaking symmetry"""
        if self.dynamic_weight:
            with torch.no_grad():
                # Get final layer (the one that outputs the 3 weights)
                final_params = list(self.weight_predictor.parameters())
                if len(final_params) >= 2:
                    final_weight = final_params[-2]  # Weight matrix of final layer
                    final_bias = final_params[-1]    # Bias vector of final layer

                    if final_weight.dim() == 2 and final_weight.size(1) == 3:
                        # Add strong asymmetric bias to break symmetry
                        asymmetric_weight = torch.tensor([
                            [0.9, -0.4, -0.5],   # Strong preference for cosine
                            [-0.5, 1.0, -0.5],   # Strong preference for covariance  
                            [-0.4, -0.6, 1.1]    # Strong preference for variance
                        ], device=final_weight.device)
                        
                        # Apply asymmetric modification
                        final_weight.data = final_weight.data * 0.3 + asymmetric_weight * 0.7

                        # Apply strong asymmetric bias
                        if final_bias.size(0) == 3:
                            asymmetric_bias = torch.tensor([0.25, -0.12, -0.13], device=final_bias.device)
                            final_bias.data = final_bias.data * 0.5 + asymmetric_bias * 0.5

                        print(f"🎛️ FORCED: Strong asymmetric bias applied for dynamic learning")

    def get_current_weight_diversity(self):
        """Get current diversity of dynamic weights"""
        if self.dynamic_weight and len(self.weight_history) > 0:
            recent_weights = self.weight_history[-10:] if len(self.weight_history) >= 10 else self.weight_history
            if recent_weights:
                weights_array = np.array(recent_weights)
                return np.std(weights_array)
        return 0.0

    def compute_regularized_covariance(self, f_q, f_k):
        """Your covariance formula with breakthrough-level stability"""
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # Enhanced numerical stability
        E_bar = f_k.mean(dim=2, keepdim=True)
        f_k_centered = f_k - E_bar
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)

        # Compute covariance with adaptive regularization
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5 + 1e-6)

        # BREAKTHROUGH: Health-adaptive regularization
        component_health = torch.std(cov_component).item()
        base_reg = self.lambda_reg / max(m, 1)

        if component_health < 1e-5:  # Component is nearly dead
            regularization_factor = base_reg * 8.0  # Stronger revival
        elif component_health < 1e-3:  # Component is weak
            regularization_factor = base_reg * 3.0  # Strong boost
        else:  # Component is healthy
            regularization_factor = base_reg  # Normal regularization

        cov_component = regularization_factor * cov_component
        cov_component = torch.clamp(cov_component, -30.0, 30.0)  # Wider range
        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """Your variance formula with breakthrough-level diversity preservation"""
        # Controlled normalization for stability
        f_q_norm = F.normalize(f_q, p=2, dim=-1, eps=1e-6)
        f_k_norm = F.normalize(f_k, p=2, dim=-1, eps=1e-6)

        # Cosine similarity with safe range
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))
        cosine_sim = torch.clamp(cosine_sim, -0.98, 0.98)

        # BREAKTHROUGH: More aggressive adaptive gamma
        if self.training:
            sim_diversity = torch.std(cosine_sim).item()
            if sim_diversity < 0.003:  # Very low diversity
                adaptive_gamma = self.gamma * 6.0  # Stronger boost
            elif sim_diversity < 0.015:  # Low diversity
                adaptive_gamma = self.gamma * 3.0  # Strong boost
            else:  # Good diversity
                adaptive_gamma = self.gamma * (1.0 + sim_diversity * 8.0)  # More aggressive scaling
        else:
            adaptive_gamma = self.gamma

        # Enhanced margin-based variance computation
        margin_values = torch.clamp(adaptive_gamma - cosine_sim, min=0.0, max=8.0)  # Wider range
        var_component = margin_values.mean(dim=-1, keepdim=True)
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))
        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # Your enhanced complex attention system
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_sim = torch.clamp(cosine_sim, -0.98, 0.98)

            # Your complex formulas with breakthrough enhancements
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # BREAKTHROUGH: Enhanced dynamic weighting
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)

                # Temperature-controlled prediction with adaptive range
                weight_temp = torch.clamp(self.weight_temperature, 0.05, 3.0)  # Wider range
                weight_logits = self.weight_predictor(qk_features)
                weights = F.softmax(weight_logits / weight_temp, dim=-1)

                # BREAKTHROUGH: More aggressive weight diversity enforcement
                min_weight = 0.05  # Allow more extreme specialization
                max_weight = 0.85  # Allow strong dominance
                weights = torch.clamp(weights, min_weight, max_weight)
                weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize

                # Track weight evolution
                if self.training:
                    current_weights = weights.detach().cpu().numpy().mean(axis=0)
                    self.weight_history.append(current_weights)
                    if len(self.weight_history) > 100:
                        self.weight_history.pop(0)
                    self.last_weights = torch.tensor(current_weights)

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, 0.1, 0.9)

            # BREAKTHROUGH: Component combination with health-aware scaling
            cosine_norm = torch.std(cosine_sim).detach() + 1e-6
            cov_norm = torch.std(cov_component).detach() + 1e-6
            var_norm = torch.std(var_component).detach() + 1e-6

            # Health-based component scaling with more aggressive factors
            cosine_health = min(cosine_norm.item(), 1.2)
            cov_health = min(cov_norm.item() * 12.0, 1.2)  # More aggressive scaling
            var_health = min(var_norm.item() * 12.0, 1.2)

            cosine_scaled = cosine_sim / cosine_norm * cosine_health
            cov_scaled = cov_component / cov_norm * 0.5 * cov_health  # Higher scaling
            var_scaled = var_component / var_norm * 0.5 * var_health

            # Combine all components
            dots = (cos_weight * cosine_scaled +
                   cov_weight * cov_scaled +
                   var_weight * var_scaled)

            # BREAKTHROUGH: More aggressive attention temperature - FULLY TENSOR SAFE
            dots_diversity = torch.std(dots)
            if dots_diversity < 5e-6:  # Attention completely collapsed
                attention_temperature = torch.tensor(0.03, device=dots.device, dtype=dots.dtype)
            elif dots_diversity < 5e-4:  # Attention weak
                attention_temperature = torch.tensor(0.15, device=dots.device, dtype=dots.dtype)
            else:  # Attention healthy
                attention_temperature = torch.tensor(0.4, device=dots.device, dtype=dots.dtype) + dots_diversity * 1.5

            # Tensor-safe clamping with wider range
            attention_temperature = torch.clamp(attention_temperature, 0.03, 4.0)
            dots = dots / attention_temperature

            out = torch.matmul(self.sm(dots), f_v)
        else:
            # Enhanced standard attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Enhanced statistics with breakthrough metrics"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:
            # Comprehensive health metrics
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
                'health_status': 'breakthrough' if weight_diversity > 0.1 else 'learning' if weight_diversity > 0.03 else 'stuck',
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
    """Enhanced cosine distance with breakthrough-level stability"""
    dots = torch.matmul(x1, x2)
    eps = 1e-8
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps
    scale = torch.matmul(norm1, norm2)
    result = torch.clamp(dots / scale, -0.98, 0.98)  # Safe numerical range
    return result
