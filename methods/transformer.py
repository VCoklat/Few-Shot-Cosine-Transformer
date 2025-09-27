# hybrid_attention_complete.py
# Memory-efficient hybrid attention mechanism for 16GB VRAM
# Automatically switches from PyTorch to NumPy computation at 40% accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union
import math

# ============================================================================
# Part 1: Memory Optimization Utilities and Base Classes
# ============================================================================

class MemoryOptimizer:
    """Utility class for GPU memory management with 16GB VRAM"""

    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0

    @staticmethod
    def optimize_batch_size(model, sample_input, max_memory_gb=14):
        """Automatically find optimal batch size for 16GB VRAM"""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        optimal_batch = 1

        for batch_size in batch_sizes:
            try:
                MemoryOptimizer.clear_cache()
                if sample_input.dim() == 4:
                    test_input = sample_input.repeat(batch_size, 1, 1, 1)
                elif sample_input.dim() == 3:
                    test_input = sample_input.repeat(batch_size, 1, 1)
                else:
                    test_input = sample_input.repeat(batch_size, 1)

                with torch.no_grad():
                    _ = model(test_input)

                memory_used = MemoryOptimizer.get_memory_usage()
                if memory_used < max_memory_gb:
                    optimal_batch = batch_size
                else:
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    break

        MemoryOptimizer.clear_cache()
        return optimal_batch

class AccuracyTracker:
    """Track training accuracy for dynamic switching"""

    def __init__(self, switch_threshold=40.0):
        self.switch_threshold = switch_threshold
        self.current_accuracy = 0.0
        self.has_switched = False
        self.accuracy_history = []

    def update(self, accuracy):
        """Update current accuracy"""
        self.current_accuracy = accuracy
        self.accuracy_history.append(accuracy)

        if not self.has_switched and accuracy >= self.switch_threshold:
            self.has_switched = True
            return True
        return False

    def should_use_numpy_method(self):
        """Check if should use numpy-based computation"""
        return self.has_switched

# ============================================================================
# Part 2: PyTorch-based Attention Mechanism (< 40% accuracy)
# ============================================================================

class TorchAttentionMechanism(nn.Module):
    """Memory-efficient PyTorch attention with gradient checkpointing"""

    def __init__(self, embed_dim, num_heads=8, use_checkpointing=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_checkpointing = use_checkpointing

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections with memory-efficient initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Memory-efficient initialization
        self._init_weights()

    def _init_weights(self):
        """Memory-efficient weight initialization"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))

    def _chunked_attention(self, q, k, v, chunk_size=512):
        """Memory-efficient chunked attention computation"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Process in chunks to avoid OOM
        outputs = []

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_idx, :]  # [batch, heads, chunk, head_dim]

            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention to values
            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)

    def forward(self, x):
        """Forward pass with memory optimization"""
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        """Forward implementation with YOUR ORIGINAL FORMULAS"""
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # YOUR ORIGINAL COVARIANCE AND VARIANCE COMPONENTS
        f_q = q.contiguous()  # [batch, heads, seq, head_dim]
        f_k = k.contiguous()

        # Calculate covariance component (YOUR ORIGINAL FORMULA)
        q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
        k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
        cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
        cov_component = cov_component / f_q.size(-1)

        # Calculate variance component (YOUR ORIGINAL FORMULA)
        # Compute variance along feature dimension
        q_var = torch.var(f_q, dim=-1, keepdim=True)  # [h, q, n, 1]
        k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)  # [h, q, 1, m]

        # Create variance-based attention
        var_component = torch.matmul(q_var, k_var)  # [h, q, n, m]
        var_component = var_component / f_q.size(-1)  # Scale like covariance

        # Combine components and apply to values
        attention_scores = cov_component + var_component
        attn_weights = F.softmax(attention_scores, dim=-1)

        # For very long sequences, use chunked attention
        if seq_len > 1024:
            out = self._chunked_attention(f_q, f_k, v)
        else:
            out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)

# ============================================================================
# Part 3: NumPy-based Attention Mechanism (>= 40% accuracy)
# ============================================================================

class NumpyAttentionMechanism:
    """NumPy-based attention mechanism for CPU computation"""

    def __init__(self, embed_dim, num_heads=8, gamma=1.0, epsilon=1e-8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize weights as numpy arrays
        self._init_weights()

    def _init_weights(self):
        """Initialize weights in NumPy"""
        scale = 1.0 / math.sqrt(self.embed_dim)
        self.q_weight = np.random.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
        self.k_weight = np.random.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
        self.v_weight = np.random.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
        self.out_weight = np.random.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)

    def variance_component(self, E, gamma, epsilon):
        """YOUR ORIGINAL VARIANCE COMPONENT FUNCTION"""
        # E shape: (m, d) where m is jumlah embedding, d adalah dimensi
        sigma = np.sqrt(np.var(E, axis=0) + epsilon)
        V = np.mean(np.maximum(0, gamma - sigma))
        return V

    def covariance_component(self, E):
        """YOUR ORIGINAL COVARIANCE COMPONENT FUNCTION"""
        # E shape: (m, d) where m adalah jumlah embedding, d adalah dimensi
        m = E.shape[0]
        E_mean = np.mean(E, axis=0)
        centered = E - E_mean
        cov = (centered.T @ centered) / (m - 1)

        # Jumlah kuadrat off-diagonal
        off_diag_sum = np.sum(cov ** 2) - np.sum(np.diag(cov) ** 2)
        # Normalisasi dengan dimensi
        C = off_diag_sum / E.shape[1]
        return C

    def forward(self, x_np):
        """Forward pass using NumPy computations"""
        # x_np shape: [batch, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x_np.shape

        # Linear projections
        q = np.dot(x_np, self.q_weight)  # [batch, seq, embed]
        k = np.dot(x_np, self.k_weight)
        v = np.dot(x_np, self.v_weight)

        # Process each batch item and head separately for memory efficiency
        outputs = []

        for b in range(batch_size):
            batch_outputs = []

            for h in range(self.num_heads):
                start_idx = h * self.head_dim
                end_idx = start_idx + self.head_dim

                # Extract head-specific projections
                q_head = q[b, :, start_idx:end_idx]  # [seq, head_dim]
                k_head = k[b, :, start_idx:end_idx]
                v_head = v[b, :, start_idx:end_idx]

                # Apply your custom components
                var_comp = self.variance_component(q_head, self.gamma, self.epsilon)
                cov_comp = self.covariance_component(q_head)

                # Combine components to create attention weights
                attention_factor = var_comp + cov_comp

                # Simple attention mechanism with your components
                scores = np.dot(q_head, k_head.T) / np.sqrt(self.head_dim)
                scores = scores * (1 + attention_factor)  # Apply your custom components

                # Softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Apply attention to values
                head_output = np.dot(attn_weights, v_head)  # [seq, head_dim]
                batch_outputs.append(head_output)

            # Concatenate all heads
            batch_concat = np.concatenate(batch_outputs, axis=1)  # [seq, embed_dim]
            outputs.append(batch_concat)

        # Stack batches and apply output projection
        output = np.stack(outputs, axis=0)  # [batch, seq, embed_dim]
        output = np.dot(output, self.out_weight)

        return output

# ============================================================================
# Part 4: Hybrid Attention Module with Dynamic Switching
# ============================================================================

class HybridAttentionModule(nn.Module):
    """Hybrid attention that switches based on accuracy threshold"""

    def __init__(self, embed_dim, num_heads=8, switch_threshold=40.0, 
                 gamma=1.0, epsilon=1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Initialize both attention mechanisms
        self.torch_attention = TorchAttentionMechanism(embed_dim, num_heads)
        self.numpy_attention = NumpyAttentionMechanism(embed_dim, num_heads, gamma, epsilon)

        # Accuracy tracker
        self.accuracy_tracker = AccuracyTracker(switch_threshold)

        # Memory optimizer
        self.memory_optimizer = MemoryOptimizer()

    def update_accuracy(self, accuracy):
        """Update accuracy and check for switching"""
        switched = self.accuracy_tracker.update(accuracy)
        if switched:
            print(f"🔄 Switching to NumPy-based attention at {accuracy:.1f}% accuracy")
            # Clear GPU memory when switching
            self.memory_optimizer.clear_cache()
        return switched

    def get_current_method(self):
        """Get current computation method"""
        return "NumPy (CPU)" if self.accuracy_tracker.should_use_numpy_method() else "PyTorch (GPU)"

    def forward(self, x):
        """Forward pass with dynamic switching"""
        # Check memory usage
        memory_before = self.memory_optimizer.get_memory_usage()

        if self.accuracy_tracker.should_use_numpy_method():
            # Use NumPy-based computation (CPU)
            device = x.device
            x_cpu = x.detach().cpu().numpy()

            # Process in smaller chunks if needed
            batch_size = x_cpu.shape[0]
            chunk_size = min(batch_size, 8)  # Process 8 samples at a time

            outputs = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = x_cpu[i:end_idx]
                chunk_output = self.numpy_attention.forward(chunk)
                outputs.append(chunk_output)

            # Combine outputs and move back to GPU
            output_np = np.concatenate(outputs, axis=0)
            output = torch.from_numpy(output_np).to(device)

        else:
            # Use PyTorch-based computation (GPU)
            output = self.torch_attention(x)

        # Memory management
        memory_after = self.memory_optimizer.get_memory_usage()
        if memory_after > 12.0:  # If using >12GB, clear cache
            self.memory_optimizer.clear_cache()

        return output

# ============================================================================
# Part 5: Training Loop with Memory Management
# ============================================================================

class MemoryEfficientTrainer:
    """Training loop with automatic memory management"""

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.memory_optimizer = MemoryOptimizer()

    def train_epoch(self, dataloader, epoch):
        """Train one epoch with memory management"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            # Clear cache every 10 batches
            if batch_idx % 10 == 0:
                self.memory_optimizer.clear_cache()

            data, targets = data.to(self.device), targets.to(self.device)

            # Mixed precision training for memory efficiency
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

            # Update accuracy in the model if it has hybrid attention
            if hasattr(self.model, 'update_accuracy'):
                current_accuracy = 100. * correct / total
                self.model.update_accuracy(current_accuracy)

            # Memory monitoring
            if batch_idx % 50 == 0:
                memory_used = self.memory_optimizer.get_memory_usage()
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'Memory: {memory_used:.2f}GB')

                # Emergency memory clearing
                if memory_used > 14.0:
                    print("⚠️  High memory usage detected, clearing cache...")
                    self.memory_optimizer.clear_cache()

        epoch_accuracy = 100. * correct / total
        epoch_loss = total_loss / len(dataloader)

        return epoch_loss, epoch_accuracy

# ============================================================================
# Part 6: Hybrid Model Creation Functions
# ============================================================================

def create_hybrid_model(embed_dim=512, num_classes=10, num_heads=8, switch_threshold=40.0):
    """Create a model with hybrid attention mechanism for general use"""

    class HybridTransformerModel(nn.Module):
        def __init__(self, embed_dim, num_classes, num_heads, switch_threshold):
            super().__init__()
            self.embedding = nn.Linear(784, embed_dim)  # Adjustable input size
            self.hybrid_attention = HybridAttentionModule(
                embed_dim, num_heads, switch_threshold=switch_threshold)
            self.norm = nn.LayerNorm(embed_dim)
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            # Flatten input if needed
            if x.dim() > 2:
                x = x.view(x.size(0), -1)

            # Embedding
            x = self.embedding(x)  # [batch, embed_dim]
            x = x.unsqueeze(1)     # [batch, 1, embed_dim] - add sequence dimension

            # Hybrid attention
            x = self.hybrid_attention(x)
            x = self.norm(x)
            x = self.dropout(x)

            # Classification
            x = x.squeeze(1)  # Remove sequence dimension
            x = self.classifier(x)

            return x

        def update_accuracy(self, accuracy):
            """Update accuracy for attention switching"""
            return self.hybrid_attention.update_accuracy(accuracy)

        def get_attention_method(self):
            """Get current attention method"""
            return self.hybrid_attention.get_current_method()

    return HybridTransformerModel(embed_dim, num_classes, num_heads, switch_threshold)

def create_hybrid_few_shot_model(embed_dim, num_classes, num_heads=8, switch_threshold=40.0):
    """Create a hybrid attention model specifically for few-shot learning"""

    class HybridFewShotModel(nn.Module):
        def __init__(self, embed_dim, num_classes, num_heads, switch_threshold):
            super().__init__()
            self.hybrid_attention = HybridAttentionModule(
                embed_dim, num_heads, switch_threshold=switch_threshold)
            self.norm = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            # x should already be in the right format [batch, seq, embed_dim]
            x = self.hybrid_attention(x)
            x = self.norm(x)
            x = self.dropout(x)
            return x

        def update_accuracy(self, accuracy):
            """Update accuracy for attention switching"""
            return self.hybrid_attention.update_accuracy(accuracy)

        def get_attention_method(self):
            """Get current attention method"""
            return self.hybrid_attention.get_current_method()

    return HybridFewShotModel(embed_dim, num_classes, num_heads, switch_threshold)

# ============================================================================
# Part 7: Utility Functions
# ============================================================================

def get_optimal_config(available_memory_gb=16):
    """Get optimal configuration for given memory"""
    if available_memory_gb >= 24:
        return {
            'max_batch_size': 128,
            'embed_dim': 1024,
            'num_heads': 16,
            'chunk_size': 1024
        }
    elif available_memory_gb >= 16:
        return {
            'max_batch_size': 64,
            'embed_dim': 512,
            'num_heads': 8,
            'chunk_size': 512
        }
    elif available_memory_gb >= 8:
        return {
            'max_batch_size': 32,
            'embed_dim': 256,
            'num_heads': 4,
            'chunk_size': 256
        }
    else:
        return {
            'max_batch_size': 16,
            'embed_dim': 128,
            'num_heads': 2,
            'chunk_size': 128
        }

def print_hybrid_info():
    """Print information about the hybrid attention system"""
    print("🚀 Hybrid Attention System")
    print("="*50)
    print("✅ PyTorch GPU computation: < 40% accuracy")
    print("   • Uses your original covariance/variance formulas")
    print("   • Memory-efficient with gradient checkpointing")
    print("   • Chunked processing for long sequences")
    print()
    print("🔄 Automatic switching at 40% accuracy")
    print()
    print("✅ NumPy CPU computation: >= 40% accuracy") 
    print("   • Uses your variance_component() function")
    print("   • Uses your covariance_component() function")
    print("   • Processes in memory-efficient chunks")
    print()
    print("💾 Memory optimization for 16GB VRAM:")
    print("   • Automatic batch size detection")
    print("   • GPU cache clearing")
    print("   • Emergency memory management")
    print("="*50)

# Version info
__version__ = "1.0.0"
__author__ = "Hybrid Attention System"

if __name__ == "__main__":
    print_hybrid_info()
    print(f"\n📦 Hybrid Attention Complete v{__version__}")
    print("📋 Import this module into your training script!")
    print("📄 Example: from hybrid_attention_complete import HybridAttentionModule")
