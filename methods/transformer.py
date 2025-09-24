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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 initial_cov_weight=0.3, initial_var_weight=0.2, dynamic_weight=False,
                 gamma=0.5, lambda_reg=0.1):  # FIXED: Reduced default parameters for stability
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight,
                             gamma=gamma,
                             lambda_reg=lambda_reg)  # Pass lambda_reg to Attention

        self.sm = nn.Softmax(dim=-2)

        # FIXED: Initialize prototype weights properly with smaller values
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1) * 0.1)

        # FIXED: Add normalization layers and dropout for stability
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.1)   # Add dropout
        )

        # FIXED: Improve final classification layer
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 1)
            )

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)

        # FIXED: Normalize features to prevent extreme values
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)

        # FIXED: Better prototype computation with normalized weights
        proto_weights = self.sm(self.proto_weight)
        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)

        x, query = z_proto, z_query
        for _ in range(self.depth):
            # FIXED: Add residual connections with proper scaling to prevent gradient explosion
            attn_out = self.ATTN(q=x, k=query, v=query)
            x = x + 0.1 * attn_out  # Scaled residual connection

            ffn_out = self.FFN(x)
            x = x + 0.1 * ffn_out   # Scaled residual connection

        # FIXED: Apply final normalization before classification
        x = F.normalize(x, p=2, dim=-1)
        scores = self.linear(x).squeeze()  # (q, n)

        # FIXED: Apply temperature scaling for better calibration
        temperature = 2.0
        scores = scores / temperature

        return scores

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, target)

        # FIXED: Add label smoothing for better generalization and prevent overconfidence
        smooth_target = torch.zeros_like(scores).scatter_(1, target.unsqueeze(1), 0.9)
        smooth_target += 0.1 / self.n_way
        smooth_loss = -torch.sum(F.log_softmax(scores, dim=1) * smooth_target, dim=1).mean()

        # Combine losses
        total_loss = 0.8 * loss + 0.2 * smooth_loss

        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)

        return acc, total_loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.3, 
                 initial_var_weight=0.2, dynamic_weight=False, gamma=0.5, lambda_reg=0.1):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        self.gamma = gamma  # FIXED: Reduced default gamma for stability
        self.lambda_reg = lambda_reg  # FIXED: Added lambda_reg for covariance regularization

        # FIXED: Simplified and more stable weighting scheme
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head // 2),  # Smaller intermediate layer
                nn.LayerNorm(dim_head // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_head // 2, 3),  # Predict 3 weights
                nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
            )
        else:
            # FIXED: Better initialization for fixed weights
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # FIXED: Better linear projections with dropout
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.Dropout(0.1)
        )

        self.output_linear = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.1)
        ) if project_out else nn.Identity()

        self.weight_history = []  # To store weights for analysis
        self.record_weights = False  # Toggle for weight recording

    def compute_regularized_covariance(self, f_q, f_k):
        """
        FIXED: Simplified and more stable covariance computation using regularization formula
        C(E) = (1/(m-1)) * sum((E_i - E_bar)(E_i - E_bar)^T) with lambda/m factor
        """
        # Get dimensions
        h, q, n, d = f_q.shape
        _, _, m, _ = f_k.shape

        # Compute mean E_bar = (1/m) * sum(E_j) for keys
        E_bar = f_k.mean(dim=2, keepdim=True)  # [h, q, 1, d]

        # Compute centered features: (E_i - E_bar)
        f_k_centered = f_k - E_bar  # [h, q, m, d]
        f_q_centered = f_q - f_q.mean(dim=-1, keepdim=True)  # Center queries too

        # FIXED: Simplified covariance computation to avoid complex matrix operations
        # Use cross-covariance between query and key features
        cov_component = torch.matmul(f_q_centered, f_k_centered.transpose(-1, -2))
        cov_component = cov_component / (d ** 0.5)  # Scale normalization

        # Apply regularization factor lambda/m
        regularization_factor = self.lambda_reg / max(m, 1)
        cov_component = regularization_factor * cov_component

        return cov_component

    def compute_margin_based_variance(self, f_q, f_k):
        """
        FIXED: Simplified margin-based variance computation
        V(E) = (1/m) * sum(max(0, gamma - sigma(E_i, c)))
        """
        # Normalize features for cosine similarity
        f_q_norm = F.normalize(f_q, p=2, dim=-1)  # [h, q, n, d]
        f_k_norm = F.normalize(f_k, p=2, dim=-1)  # [h, q, m, d]

        # Compute cosine similarity: sigma(E_i, c) in the formula
        cosine_sim = torch.matmul(f_q_norm, f_k_norm.transpose(-1, -2))  # [h, q, n, m]

        # FIXED: Clamp cosine similarity to prevent extreme values
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Apply the margin-based variance formula: max(0, gamma - sigma(E_i, c))
        margin_values = torch.clamp(self.gamma - cosine_sim, min=0.0)  # [h, q, n, m]

        # FIXED: Simplified variance computation - take mean and apply scaling
        var_component = margin_values.mean(dim=-1, keepdim=True)  # [h, q, n, 1]
        var_component = var_component.expand(-1, -1, -1, cosine_sim.size(-1))  # [h, q, n, m]

        return var_component

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))

        if self.variant == "cosine":
            # FIXED: Stable cosine similarity computation
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            # Clamp to prevent extreme values
            cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

            # FIXED: Use improved covariance and variance computations
            cov_component = self.compute_regularized_covariance(f_q, f_k)
            var_component = self.compute_margin_based_variance(f_q, f_k)

            if self.dynamic_weight:
                # Simplified dynamic weighting
                q_global = f_q.mean(dim=(1, 2))  # [h, d]
                k_global = f_k.mean(dim=(1, 2))  # [h, d]
                qk_features = torch.cat([q_global, k_global], dim=-1)  # [h, 2d]

                weights = self.weight_predictor(qk_features)  # [h, 3]

                # Record weights during evaluation if needed
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))

                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
            else:
                # FIXED: Stable fixed weighting with proper constraints
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                # Ensure weights sum to approximately 1 and are in reasonable range
                cos_weight = 1.0 - cov_weight - var_weight
                cos_weight = torch.clamp(cos_weight, 0.1, 0.8)  # Prevent extreme values

            # FIXED: Stable combination with proper scaling
            dots = (cos_weight * cosine_sim +
                   cov_weight * cov_component * 0.1 +  # Scale down covariance
                   var_weight * var_component * 0.1)   # Scale down variance

            # FIXED: Apply temperature scaling to prevent extreme attention weights
            dots = dots / 2.0

            # FIXED: Apply softmax with numerical stability
            attention_weights = self.sm(dots)
            out = torch.matmul(attention_weights, f_v)
        else:  # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Returns statistics about the weights used"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:  # We have 3 components
            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=10, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=10, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=10, range=(0,1))[0].tolist()
                }
            }
        else:  # Legacy format with single weight
            weights = np.array(self.weight_history)
            return {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'min': float(weights.min()),
                'max': float(weights.max()),
                'histogram': np.histogram(weights, bins=10, range=(0,1))[0].tolist()
            }

    def clear_weight_history(self):
        """Clear recorded weights"""
        self.weight_history = []

def cosine_distance(x1, x2):
    """FIXED: More stable cosine distance computation with numerical stability"""
    # x1 = [b, h, n, k]
    # x2 = [b, h, k, m] 
    # output = [b, h, n, m]
    dots = torch.matmul(x1, x2)

    # Compute norms with small epsilon for numerical stability
    eps = 1e-8
    norm1 = torch.norm(x1, 2, dim=-1, keepdim=True) + eps
    norm2 = torch.norm(x2, 2, dim=-2, keepdim=True) + eps

    # Compute scale with proper broadcasting
    scale = torch.matmul(norm1, norm2)

    # Clamp the result to prevent extreme values
    result = torch.clamp(dots / scale, -1 + eps, 1 - eps)
    return result

# FIXED: Debugging and utility functions
def debug_model_predictions(model, test_loader, device='cuda', max_episodes=5):
    """Debug function to analyze model predictions and identify issues"""
    model.eval()
    print("🔍 DEBUGGING MODEL PREDICTIONS:")
    print("=" * 50)

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= max_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)

            # Get targets
            n_way = scores.size(1)
            n_query = scores.size(0) // n_way
            target = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)

            predictions = torch.argmax(scores, dim=1)

            print(f"Episode {i+1}:")
            print(f"  📊 Scores shape: {scores.shape}")
            print(f"  📈 Score range: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  📉 Score std: {scores.std():.3f}")
            print(f"  🎯 Predictions: {predictions.cpu().numpy()}")
            print(f"  ✅ Targets: {target.cpu().numpy()}")
            print(f"  🔢 Unique predictions: {torch.unique(predictions).cpu().numpy()}")
            print(f"  ✔️  Accuracy: {(predictions == target).float().mean():.3f}")
            print()

            # Check for problematic patterns
            if len(torch.unique(predictions)) < n_way:
                print(f"  ⚠️  WARNING: Only predicting {len(torch.unique(predictions))} out of {n_way} classes!")

            if torch.std(scores) < 0.1:
                print(f"  ⚠️  WARNING: Scores have very low variance ({scores.std():.3f})")

            if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
                print(f"  ❌ ERROR: NaN or Inf values detected in scores!")

    print("=" * 50)

def quick_accuracy_test(model, test_loader, device='cuda', n_episodes=10):
    """Quick test to verify the model is working properly"""
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(5)
    class_total = torch.zeros(5)

    print("🧪 QUICK ACCURACY TEST:")
    print("=" * 30)

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= n_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)
            pred = torch.argmax(scores, dim=1)

            n_way = scores.size(1)
            n_query = scores.size(0) // n_way
            target = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)

            correct += (pred == target).sum().item()
            total += target.size(0)

            # Per-class accuracy
            for j in range(min(n_way, 5)):
                mask = (target == j)
                if mask.sum() > 0:
                    class_correct[j] += (pred[mask] == target[mask]).sum().item()
                    class_total[j] += mask.sum().item()

    overall_acc = 100 * correct / total
    print(f"📈 Overall Accuracy: {overall_acc:.2f}%")
    print("📊 Per-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"   Class {i}: {class_acc:.2f}%")
        else:
            print(f"   Class {i}: No samples")

    # Health check
    if overall_acc > 25:  # Better than random for 5-way
        print("✅ Model appears to be working!")
    elif overall_acc > 15:
        print("⚠️  Model is learning but still has issues")
    else:
        print("❌ Model is not learning - still at random chance")

    print("=" * 30)
    return overall_acc / 100

print("✅ FIXED TRANSFORMER CODE GENERATED!")
print("\n🔧 Key Fixes Applied:")
print("1. ✅ Reduced default parameters (gamma=0.5, lambda_reg=0.1)")
print("2. ✅ Better prototype weight initialization (0.1 instead of 1.0)")
print("3. ✅ Added feature normalization to prevent extreme values")
print("4. ✅ Scaled residual connections (0.1x) to prevent gradient explosion")
print("5. ✅ Added dropout layers for better generalization")
print("6. ✅ Implemented label smoothing to prevent overconfidence")
print("7. ✅ Added temperature scaling for better calibration")
print("8. ✅ Improved numerical stability in all computations")
print("9. ✅ Added comprehensive debugging tools")
print("10. ✅ Simplified complex mathematical operations for stability")
