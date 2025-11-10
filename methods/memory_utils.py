"""
Memory optimization utilities for Few-Shot Learning
Helps prevent OOM (Out-of-Memory) errors on GPUs with limited VRAM (e.g., 8GB)
"""

import torch
import gc
import warnings


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def print_gpu_memory():
    """Print GPU memory information"""
    info = get_gpu_memory_info()
    if info:
        print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
              f"Reserved: {info['reserved_gb']:.2f}GB, "
              f"Max Allocated: {info['max_allocated_gb']:.2f}GB")
    else:
        print("GPU not available")


def enable_memory_efficient_mode():
    """
    Enable memory-efficient settings for PyTorch
    Useful for training on GPUs with limited VRAM (e.g., 8GB)
    """
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        
        # Use deterministic algorithms when possible
        torch.use_deterministic_algorithms(False)
        
        # Clear cache
        clear_gpu_memory()
        
        print("Memory-efficient mode enabled:")
        print("  - TF32 enabled for matrix operations")
        print("  - cuDNN auto-tuner enabled")
        print("  - GPU cache cleared")
    else:
        print("GPU not available, memory optimizations skipped")


def check_oom_risk(model, batch_size, image_size=224):
    """
    Estimate memory requirements and warn if OOM risk is high
    
    Args:
        model: PyTorch model
        batch_size: Training batch size
        image_size: Input image size
    """
    if not torch.cuda.is_available():
        return
    
    # Get available GPU memory
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)  # GB
    
    # Estimate activation memory (rough estimate)
    # Assume 4 bytes per float32 value
    activation_estimate = (batch_size * 3 * image_size * image_size * 4) / (1024 ** 3)  # GB
    
    # Total estimated memory
    estimated_memory = param_size * 2 + activation_estimate * 2  # Factor of 2 for gradients
    
    print(f"Memory Estimate:")
    print(f"  - Total GPU Memory: {total_memory:.2f}GB")
    print(f"  - Model Parameters: {param_size:.2f}GB")
    print(f"  - Estimated Total Usage: {estimated_memory:.2f}GB")
    
    # Warn if estimated memory is close to or exceeds available memory
    if estimated_memory > total_memory * 0.8:
        warnings.warn(
            f"⚠ High OOM Risk! Estimated memory usage ({estimated_memory:.2f}GB) "
            f"is close to or exceeds available GPU memory ({total_memory:.2f}GB). "
            "Consider reducing batch size or using gradient checkpointing.",
            UserWarning
        )
        return True
    elif estimated_memory > total_memory * 0.6:
        print(f"⚠ Moderate OOM risk. Consider monitoring GPU memory usage.")
        return False
    else:
        print(f"✓ Memory usage looks safe.")
        return False


def setup_mixed_precision_training():
    """
    Setup automatic mixed precision (AMP) training for memory efficiency
    Returns a GradScaler for AMP training
    """
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training (AMP) enabled for memory efficiency")
        return scaler
    else:
        print("GPU not available, mixed precision training not enabled")
        return None


class MemoryMonitor:
    """Context manager for monitoring memory usage"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_memory = 0
        
    def __enter__(self):
        clear_gpu_memory()
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / (1024 ** 2)  # MB
            print(f"{self.name}: Used {memory_used:.2f}MB of GPU memory")


def suggest_batch_size_reduction(current_batch_size):
    """Suggest a reduced batch size to avoid OOM"""
    suggested = max(1, current_batch_size // 2)
    print(f"Suggestion: Try reducing batch size from {current_batch_size} to {suggested}")
    return suggested


# Gradient checkpointing wrapper for memory efficiency
def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for compatible models to save memory
    Trade compute for memory by not storing intermediate activations
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
        return True
    else:
        print("Model does not support gradient checkpointing")
        return False


# Memory-efficient training tips
MEMORY_TIPS = """
Memory Optimization Tips for Few-Shot Learning:
1. Reduce batch size (n_episode parameter)
2. Use smaller backbone (Conv4 instead of ResNet34)
3. Reduce image size (84 instead of 224 for ResNet)
4. Enable mixed precision training (AMP)
5. Reduce k_shot or n_way if possible
6. Use gradient accumulation for larger effective batch sizes
7. Clear cache between training steps
8. Use VIC regularization with dynamic weights (automatically adapts)
"""


def print_memory_tips():
    """Print memory optimization tips"""
    print(MEMORY_TIPS)
