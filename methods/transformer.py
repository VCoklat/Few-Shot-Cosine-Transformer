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
    """EMERGENCY: Ultra-aggressive intervention tracking"""
    def __init__(self):
        self.variance_history = []
        self.intervention_cooldown = 0
        self.stuck_counter = 0
        self.last_intervention_variance = 0.0

    def should_intervene(self, current_variance, batch_idx):
        """EMERGENCY: Ultra-aggressive intervention decision"""
        # Always track variance history
        self.variance_history.append(current_variance)
        if len(self.variance_history) > 15:  # Shorter history
            self.variance_history.pop(0)

        # Reduce cooldown
        if self.intervention_cooldown > 0:
            self.intervention_cooldown -= 1
            return False, "cooling_down"

        # EMERGENCY: Trigger on ANY low variance
        if current_variance < 0.002:  # Much lower threshold
            self.intervention_cooldown = 3  # Very short cooldown
            return True, "emergency"

        # EMERGENCY: Trigger if no improvement over very short period
        if len(self.variance_history) >= 6:  # Much shorter window
            recent_vars = self.variance_history[-3:]
            old_vars = self.variance_history[-6:-3]
            recent_mean = np.mean(recent_vars)
            old_mean = np.mean(old_vars)

            if abs(recent_mean - old_mean) < 0.0003:  # Much smaller threshold
                self.intervention_cooldown = 3
                return True, "truly_stuck"

        # EMERGENCY: Always trigger if completely flat
        if current_variance < 0.0005:
            self.intervention_cooldown = 2
            return True, "critical"

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

        # EMERGENCY: Ultra-aggressive intervention tracking
        self.intervention_tracker = InterventionTracker()
        self.training_epoch = 0
        self.batch_count = 0
        self.major_interventions = 0
        self.last_variance = 0.0
        self.breakthrough_achieved = False
        self.emergency_mode = True  # Start in emergency mode

        # EMERGENCY: Enhanced attention with all breakthrough fixes
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight,
                             gamma=gamma,
                             lambda_reg=lambda_reg)

        # EMERGENCY: Ultra-aggressive temperature system
        self.sm = nn.Softmax(dim=-2)
        self.temperature_sm = nn.Parameter(torch.ones(1) * 10.0)  # Start very high

        # EMERGENCY: More aggressive initialization for breakthrough
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.25)  # Higher variance

        # EMERGENCY: Simplified FFN with minimal dropout
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.05),  # Minimal dropout
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.05)   # Minimal dropout
        )

        # EMERGENCY: Simplified classification layers
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.05),  # Minimal dropout
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Dropout(0.05),  # Minimal dropout
                nn.Linear(dim_head, 1)
            )

    def emergency_nuclear_reset(self):
        """EMERGENCY: Complete nuclear reset of all systems"""
        print("🚨 NUCLEAR RESET: Complete system reinitialization")
        
        with torch.no_grad():
            # 1. Reset attention system
            if hasattr(self.ATTN, 'nuclear_breakthrough'):
                self.ATTN.nuclear_breakthrough()
            
            # 2. Reset temperature to maximum exploration
            self.temperature_sm.data.fill_(20.0)
            
            # 3. Massive proto weight shake-up
            noise = torch.randn_like(self.proto_weight) * 1.0  # Massive noise
            self.proto_weight.data = noise
            
            # 4. Reset FFN parameters
            for module in self.FFN.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=2.0)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.1, 0.1)
            
            # 5. Reset classification layer
            for module in self.linear.modules():
                if isinstance(module, (nn.Linear, CosineDistLinear)):
                    if hasattr(module, 'weight'):
                        nn.init.xavier_uniform_(module.weight, gain=3.0)  # Higher gain
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.uniform_(module.bias, -0.2, 0.2)
        
        print("☢️ NUCLEAR RESET COMPLETE")

    def breakthrough_intervention(self, current_variance, intervention_type):
        """EMERGENCY: Immediate and aggressive interventions"""
        self.major_interventions += 1
        
        if intervention_type in ["emergency", "critical", "truly_stuck"]:
            print(f"\n🚨 EMERGENCY INTERVENTION #{self.major_interventions}")
            print(f"   Type: {intervention_type.upper()}")
            print(f"   Current variance: {current_variance:.8f}")
            
            # EMERGENCY: Always apply nuclear reset
            self.emergency_nuclear_reset()
            
            # EMERGENCY: Force dynamic weight breakthrough
            if hasattr(self.ATTN, 'force_dynamic_learning'):
                self.ATTN.force_dynamic_learning()
            
            # EMERGENCY: Set breakthrough achieved if any improvement
            if current_variance > 0.001:
                self.breakthrough_achieved = True
                print(f"   🎉 BREAKTHROUGH ACHIEVED!")
            
            return True
        
        return False

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # EMERGENCY: Minimal normalization to preserve diversity
        if self.emergency_mode:
            norm_strength = 0.3  # Much lighter normalization
        else:
            norm_strength = 0.6
            
        z_support = F.normalize(z_support, p=2, dim=-1) * norm_strength + z_support * (1 - norm_strength)
        z_query = F.normalize(z_query, p=2, dim=-1) * norm_strength + z_query * (1 - norm_strength)

        # EMERGENCY: Aggressive temperature control
        if self.training:
            temp_min, temp_max = 0.05, 25.0
        else:
            temp_min, temp_max = 0.1, 5.0
            
        current_temp = torch.clamp(self.temperature_sm, min=temp_min, max=temp_max)

        # Prototype computation with emergency diversity enforcement
        proto_logits = self.proto_weight / current_temp
        proto_weights = self.sm(proto_logits)

        # EMERGENCY: Reduce uniform weighting more aggressively
        if self.breakthrough_achieved:
            uniform_weight = 0.03  # Very little uniform weighting
        elif self.emergency_mode:
            uniform_weight = 0.15  # Moderate uniform weighting in emergency
        else:
            uniform_weight = 0.25  # Normal uniform weighting

        proto_weights = proto_weights * (1 - uniform_weight) + uniform_weight / self.k_shot
        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)
        x, query = z_proto, z_query

        # EMERGENCY: Simplified attention processing
        for layer_idx in range(self.depth):
            x_input = x.clone()
            attn_out = self.ATTN(q=x, k=query, v=query)

            # EMERGENCY: Ultra-aggressive variance-adaptive residual scaling
            if self.training:
                if self.last_variance < 0.0005:  # Critical variance
                    variance_boost = 15.0  # Massive boost
                elif self.last_variance < 0.002:  # Low variance
                    variance_boost = 8.0   # Large boost
                else:  # Healthy variance
                    variance_boost = 1.0   # Normal scaling

                scale_factor = 0.3 * variance_boost  # Higher base scaling
                scale_factor = min(scale_factor, 1.5)  # Higher cap
            else:
                scale_factor = 0.25

            x = x_input + scale_factor * attn_out

            # EMERGENCY: Simplified FFN processing
            ffn_out = self.FFN(x.view(-1, x.size(-1))).view(x.shape)
            ffn_scale = 0.4 if self.breakthrough_achieved else 0.3
            x = x + ffn_scale * ffn_out

        # EMERGENCY: Simplified final processing
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=0.001, max=100.0)

        scores = self.linear(x).squeeze()

        # EMERGENCY: Ultra-aggressive score diversity system
        if self.training:
            score_std = torch.std(scores)
            score_range = scores.max() - scores.min()
            
            # EMERGENCY: More aggressive diversity enforcement
            if score_std < 0.1:  # Very low threshold
                noise_scale = max(0.4, 0.3 / (score_std + 1e-8))
                noise_scale = min(noise_scale, 3.0)  # Higher cap
                noise = torch.randn_like(scores) * noise_scale
                scores = scores + noise

                if score_std < 0.03:  # Critical
                    print(f"🚨 EMERGENCY SCORE DIVERSITY: std={score_std:.6f}, range={score_range:.6f}")

        # EMERGENCY: Ultra-aggressive final temperature scaling
        if self.training:
            score_range = scores.max() - scores.min()
            if score_range < 0.05:  # Very compressed
                final_temp = torch.tensor(0.02, device=scores.device, dtype=scores.dtype)
            elif score_range > 20.0:  # Over-spread
                final_temp = torch.tensor(5.0, device=scores.device, dtype=scores.dtype)
            else:
                final_temp = torch.tensor(0.8, device=scores.device, dtype=scores.dtype)
        else:
            final_temp = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

        scores = scores / final_temp
        return scores

    def set_forward_loss(self, x):
        """EMERGENCY: Simplified loss with minimal regularization"""
        self.batch_count += 1
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)
        current_variance = torch.var(scores).item()
        self.last_variance = current_variance

        # EMERGENCY: Ultra-aggressive intervention system
        should_intervene, intervention_type = self.intervention_tracker.should_intervene(
            current_variance, self.batch_count)

        if should_intervene and intervention_type != "cooling_down":
            intervention_applied = self.breakthrough_intervention(current_variance, intervention_type)
            if intervention_applied:
                # Re-compute scores after intervention
                scores = self.set_forward(x)
                new_variance = torch.var(scores).item()
                print(f"   📊 Variance change: {current_variance:.8f} → {new_variance:.8f}")
                current_variance = new_variance
                self.last_variance = current_variance

        # EMERGENCY: Simplified base loss
        base_loss = self.loss_fn(scores, target)

        # EMERGENCY: Remove ALL regularization for first 15 epochs
        if self.training_epoch < 15:
            total_loss = base_loss
            if self.batch_count % 50 == 0:
                print(f"🚨 EMERGENCY MODE: Pure base loss = {base_loss:.6f}")
        else:
            # EMERGENCY: Minimal regularization after basic learning
            if current_variance < 0.001:
                # Only add minimal entropy bonus when really needed
                probs = F.softmax(scores, dim=1)
                score_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                entropy_bonus = 0.05 * score_entropy  # Very weak
                total_loss = base_loss - entropy_bonus
            else:
                total_loss = base_loss

        # EMERGENCY: Update emergency mode status
        if current_variance > 0.005 and self.training_epoch > 10:
            if self.emergency_mode:
                print("🎉 EXITING EMERGENCY MODE - Variance breakthrough achieved!")
                self.emergency_mode = False
            self.breakthrough_achieved = True

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, total_loss

class Attention(nn.Module):
    """EMERGENCY: Simplified but powerful attention with breakthrough systems"""
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

        # EMERGENCY: Ultra-powerful dynamic weighting system
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head * 4),  # Much larger
                nn.LayerNorm(dim_head * 4),
                nn.ReLU(),
                nn.Dropout(0.05),  # Minimal dropout
                nn.Linear(dim_head * 4, dim_head * 3),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(dim_head * 3, dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 3),
                nn.Softmax(dim=-1)
            )
            self.weight_temperature = nn.Parameter(torch.ones(1) * 0.1)  # Very sensitive
            self.weight_history = []
            # EMERGENCY: Start with highly asymmetric weights
            self.last_weights = torch.tensor([0.6, 0.1, 0.3])  # Very asymmetric
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # EMERGENCY: Simplified processing with minimal dropout
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.05)  # Minimal dropout
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.05)
        ) if project_out else nn.Identity()

        self.record_weights = False

    def nuclear_breakthrough(self):
        """EMERGENCY: Complete nuclear reset of weight predictor"""
        if self.dynamic_weight:
            print("☢️ NUCLEAR: Complete weight predictor reset")
            with torch.no_grad():
                # EMERGENCY: Complete reinitialization with extreme asymmetry
                for i, module in enumerate(self.weight_predictor.modules()):
                    if isinstance(module, nn.Linear):
                        # Initialize with extreme asymmetric bias
                        nn.init.xavier_uniform_(module.weight, gain=3.0)
                        if module.bias is not None:
                            if i == 0:  # First layer - favor asymmetry
                                module.bias.uniform_(-0.1, 0.1)
                            elif module.out_features == 3:  # Final layer - extreme asymmetry
                                module.bias.data = torch.tensor(
                                    [0.5, -0.3, -0.2], device=module.bias.device)
                            else:
                                module.bias.uniform_(-0.05, 0.05)

                # EMERGENCY: Very sensitive temperature
                self.weight_temperature.data.fill_(0.05)

                # Clear history
                self.weight_history = []
                print("☢️ NUCLEAR: Complete reset applied with extreme asymmetry")

    def emergency_diversity_injection(self):
        """EMERGENCY: Massive parameter diversity injection"""
        if self.dynamic_weight:
            print("💥 EMERGENCY: Massive diversity injection")
            with torch.no_grad():
                # EMERGENCY: Inject massive controlled noise
                for param in self.weight_predictor.parameters():
                    noise_scale = min(0.5, max(0.1, param.std().item()))
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)

                # EMERGENCY: Make temperature ultra-sensitive
                self.weight_temperature.data.fill_(0.02)
                print(f"💥 EMERGENCY: Massive noise applied, temp set to 0.02")

    def force_dynamic_learning(self):
        """EMERGENCY: Force extreme asymmetric learning"""
        if self.dynamic_weight:
            print("🎛️ FORCE: Extreme asymmetric weight forcing")
            with torch.no_grad():
                # Get final layer parameters
                final_layers = [m for m in self.weight_predictor.modules() if isinstance(m, nn.Linear)]
                if final_layers:
                    final_layer = final_layers[-1]
                    
                    if final_layer.out_features == 3:
                        # EMERGENCY: Force extreme asymmetric weights
                        extreme_weight = torch.tensor([
                            [2.0, -1.5, -0.5],   # Extreme cosine preference
                            [-1.0, 2.5, -1.5],   # Extreme covariance preference  
                            [-0.5, -2.0, 3.0]    # Extreme variance preference
                        ], device=final_layer.weight.device)
                        
                        final_layer.weight.data = extreme_weight
                        
                        if final_layer.bias is not None:
                            final_layer.bias.data = torch.tensor(
                                [0.8, -0.4, -0.4], device=final_layer.bias.device)

                        print("🎛️ FORCE: Extreme asymmetric bias applied")

    def get_current_weight_diversity(self):
        """Get current diversity of dynamic weights"""
        if self.dynamic_weight and len(self.weight_history) > 0:
            recent_weights = self.weight_history[-5:] if len(self.weight_history) >= 5 else self.weight_history
            if recent_weights:
                weights_array = np.array(recent_weights)
                return np.std(weights_array)
        return 0.0

    def compute_regularized_covariance(self, f_q, f_k):
        """EMERGENCY: Simplified covariance with breakthrough stability"""
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # EMERGENCY: Simplified centering
        f_k_centered = f_k - f_k.mean(dim=2, keepdim=True)
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)

        # EMERGENCY: Simple covariance computation
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5 + 1e-4)

        # EMERGENCY: Minimal regularization
        regularization_factor = self.lambda_reg / max(m, 1)
        cov_component = regularization_factor * cov_component
        cov_component = torch.clamp(cov_component, -50.0, 50.0)
        
        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """EMERGENCY: Simplified variance with breakthrough diversity"""
        # EMERGENCY: Simple normalization
        f_q_norm = F.normalize(f_q, p=2, dim=-1, eps=1e-4)
        f_k_norm = F.normalize(f_k, p=2, dim=-1, eps=1e-4)

        # EMERGENCY: Simple cosine similarity
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))
        cosine_sim = torch.clamp(cosine_sim, -0.95, 0.95)

        # EMERGENCY: Aggressive adaptive gamma
        if self.training:
            sim_std = torch.std(cosine_sim).item()
            if sim_std < 0.001:  # Very low diversity
                adaptive_gamma = self.gamma * 20.0  # Massive boost
            elif sim_std < 0.01:   # Low diversity
                adaptive_gamma = self.gamma * 10.0  # Large boost
            else:  # Good diversity
                adaptive_gamma = self.gamma * (1.0 + sim_std * 15.0)
        else:
            adaptive_gamma = self.gamma

        # EMERGENCY: Simple margin-based variance
        margin_values = torch.clamp(adaptive_gamma - cosine_sim, min=0.0, max=15.0)
        var_component = margin_values.mean(dim=-1, keepdim=True)
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))
        
        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # EMERGENCY: Simplified attention computation
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_sim = torch.clamp(cosine_sim, -0.95, 0.95)

            # Your complex formulas with emergency simplifications
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # EMERGENCY: Simplified dynamic weighting
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)

                # EMERGENCY: Ultra-sensitive temperature control
                weight_temp = torch.clamp(self.weight_temperature, 0.01, 1.0)
                weight_logits = self.weight_predictor(qk_features)
                weights = F.softmax(weight_logits / weight_temp, dim=-1)

                # EMERGENCY: Allow extreme specialization
                min_weight = 0.01  # Very low minimum
                max_weight = 0.95  # Very high maximum
                weights = torch.clamp(weights, min_weight, max_weight)
                weights = weights / weights.sum(dim=-1, keepdim=True)

                # EMERGENCY: Track weight evolution more aggressively
                if self.training:
                    current_weights = weights.detach().cpu().numpy().mean(axis=0)
                    self.weight_history.append(current_weights)
                    if len(self.weight_history) > 50:  # Shorter history
                        self.weight_history.pop(0)
                    self.last_weights = torch.tensor(current_weights)

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, 0.1, 0.9)

            # EMERGENCY: Simplified component combination
            cosine_norm = torch.std(cosine_sim).detach() + 1e-4
            cov_norm = torch.std(cov_component).detach() + 1e-4
            var_norm = torch.std(var_component).detach() + 1e-4

            # EMERGENCY: Simple health-based scaling
            cosine_scaled = cosine_sim / cosine_norm
            cov_scaled = cov_component / cov_norm * 0.3
            var_scaled = var_component / var_norm * 0.3

            # Combine all components
            dots = (cos_weight * cosine_scaled +
                   cov_weight * cov_scaled +
                   var_weight * var_scaled)

            # EMERGENCY: Ultra-aggressive attention temperature
            dots_std = torch.std(dots)
            if dots_std < 1e-5:  # Completely collapsed
                attention_temperature = torch.tensor(0.01, device=dots.device, dtype=dots.dtype)
            elif dots_std < 1e-3:  # Weak
                attention_temperature = torch.tensor(0.05, device=dots.device, dtype=dots.dtype)
            else:  # Healthy
                attention_temperature = torch.tensor(0.2, device=dots.device, dtype=dots.dtype) + dots_std * 2.0

            attention_temperature = torch.clamp(attention_temperature, 0.01, 8.0)
            dots = dots / attention_temperature

            out = torch.matmul(self.sm(dots), f_v)
        else:
            # EMERGENCY: Simple standard attention
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
            # EMERGENCY: Simplified metrics
            weight_diversity = np.std(weights, axis=0).sum()
            weight_balance = 1.0 - np.std(weights.mean(axis=0)) / 0.577
            
            learning_trend = 0.0
            if len(weights) >= 5:  # Shorter window
                recent_diversity = np.std(weights[-5:], axis=0).sum()
                old_diversity = np.std(weights[:-5], axis=0).sum() if len(weights) > 5 else 0
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
                'health_status': 'breakthrough' if weight_diversity > 0.05 else 'learning' if weight_diversity > 0.01 else 'stuck',
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=5, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=5, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=5, range=(0,1))[0].tolist()
                }
            }
        return {}

    def clear_weight_history(self):
        # EMERGENCY: Keep some history for continuity
        if len(self.weight_history) > 10:
            self.weight_history = self.weight_history[-5:]  # Keep last 5
        else:
            self.weight_history = []

def cosine_distance(x1, x2):
    """EMERGENCY: Ultra-stable cosine distance"""
    dots = torch.matmul(x1, x2)
    eps = 1e-6  # Slightly larger epsilon
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps
    scale = torch.matmul(norm1, norm2)
    result = torch.clamp(dots / scale, -0.95, 0.95)  # Safer numerical range
    return result
