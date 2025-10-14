"""
Example: Using Dynamic Weighting in Training

This example shows how to use the dynamic weighting feature in both
FewShotTransformer and CTX models.
"""

# Example 1: Using FewShotTransformer with dynamic weighting
print("="*70)
print("Example 1: FewShotTransformer with Dynamic Weighting")
print("="*70)

code_example_1 = """
from methods.transformer import FewShotTransformer
import backbone

# Define model function
def model_func():
    return backbone.ResNet10(flatten=False)

# Create model WITH regularization (recommended for better accuracy)
model = FewShotTransformer(
    model_func=model_func,
    n_way=5,           # 5-way classification
    k_shot=5,          # 5-shot learning
    n_query=15,        # 15 query samples per class
    variant="softmax", # or "cosine"
    depth=1,
    heads=8,
    dim_head=64,
    mlp_dim=512,
    # NEW PARAMETERS for dynamic weighting:
    gamma=0.1,                # Variance target (0.1 as per paper)
    epsilon=1e-8,             # Numerical stability
    use_regularization=True   # Enable dynamic weighting
)

# Training loop (simplified)
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # set_forward_loss now computes:
        # loss = w_ce * CrossEntropy + w_var * Variance + w_cov * Covariance
        # where weights are dynamically predicted
        acc, loss = model.set_forward_loss(batch)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Acc: {acc:.4f}, Loss: {loss.item():.4f}")
"""

print(code_example_1)

# Example 2: Using CTX with dynamic weighting
print("\n" + "="*70)
print("Example 2: CTX (CrossTransformer) with Dynamic Weighting")
print("="*70)

code_example_2 = """
from methods.CTX import CTX
import backbone

# Define model function
def model_func():
    return backbone.ResNet10(flatten=False)

# Create CTX model WITH regularization
model = CTX(
    model_func=model_func,
    n_way=5,
    k_shot=5,
    n_query=15,
    heatmap=0,
    variant="softmax",
    input_dim=64,
    dim_attn=128,
    # NEW PARAMETERS for dynamic weighting:
    gamma=0.1,                # Variance target
    epsilon=1e-8,             # Numerical stability
    use_regularization=True   # Enable dynamic weighting
)

# The rest of training is the same as Example 1
"""

print(code_example_2)

# Example 3: Disabling regularization (baseline comparison)
print("\n" + "="*70)
print("Example 3: Baseline Model WITHOUT Regularization")
print("="*70)

code_example_3 = """
# To compare performance, you can disable regularization:

model_baseline = FewShotTransformer(
    model_func=model_func,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="softmax",
    use_regularization=False  # Disable regularization
)

# This will use only cross-entropy loss, like the original implementation
"""

print(code_example_3)

# Example 4: Customizing regularization parameters
print("\n" + "="*70)
print("Example 4: Custom Regularization Parameters")
print("="*70)

code_example_4 = """
# You can experiment with different gamma values:

model_custom = FewShotTransformer(
    model_func=model_func,
    n_way=5,
    k_shot=5,
    n_query=15,
    gamma=0.05,              # Lower variance target (more strict)
    epsilon=1e-6,            # Different epsilon
    use_regularization=True
)

# Or use higher gamma for less strict variance constraint:
model_lenient = FewShotTransformer(
    model_func=model_func,
    n_way=5,
    k_shot=5,
    n_query=15,
    gamma=0.2,               # Higher variance target (less strict)
    use_regularization=True
)
"""

print(code_example_4)

# Example 5: Integration with existing training script
print("\n" + "="*70)
print("Example 5: Integration with train.py")
print("="*70)

code_example_5 = """
# In your train.py, simply modify the model instantiation:

# OLD CODE:
# model = FewShotTransformer(backbone.ResNet10, n_way=5, k_shot=5, n_query=15)

# NEW CODE (with dynamic weighting):
model = FewShotTransformer(
    backbone.ResNet10, 
    n_way=5, 
    k_shot=5, 
    n_query=15,
    gamma=0.1,                # Add this
    epsilon=1e-8,             # Add this
    use_regularization=True   # Add this
)

# Everything else in your training script remains the same!
# The model.train_loop() and model.val_loop() will automatically use
# the new dynamic weighting loss.
"""

print(code_example_5)

print("\n" + "="*70)
print("Benefits of Dynamic Weighting:")
print("="*70)
print("""
1. IMPROVED ACCURACY: Adaptive combination of three loss objectives
2. BETTER GENERALIZATION: Regularization prevents overfitting
3. STABLE TRAINING: Variance constraint helps with numerical stability
4. BACKWARD COMPATIBLE: Set use_regularization=False for original behavior
5. AUTOMATIC WEIGHTING: No manual tuning of loss component weights needed
""")

print("="*70)
print("Key Parameters:")
print("="*70)
print("""
gamma (float, default=0.1):
    Target variance for regularization. Lower values enforce stricter
    variance constraints. Paper uses 0.1.

epsilon (float, default=1e-8):
    Small constant for numerical stability in variance computation.

use_regularization (bool, default=True):
    Enable/disable the dynamic weighting. Set to False to revert to
    original cross-entropy only loss.
""")

print("="*70)
print("For more details, see DYNAMIC_WEIGHTING.md")
print("="*70)
