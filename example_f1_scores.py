"""
Example script showing how F1 scores are displayed during training/validation.

This example demonstrates the new F1 score functionality added to the validation loop.
When you run training, you'll now see per-class F1 scores after each validation epoch.
"""

import sys
import os

# This is just documentation - showing what the output looks like
print("=" * 70)
print("F1 Scores in Few-Shot Cosine Transformer")
print("=" * 70)

print("""
## During Training

When you run train.py, the validation loop will now display F1 scores
after each epoch:

Example output:
--------------

Epoch 001/100:
  Training...
  [████████████████████████████████] 100% | Acc 78.45% | Loss 0.532

  Validation...
  [████████████████████████████████] 100% | Acc 85.32%
  
  Val Acc = 85.32% +- 1.24%

  📊 Validation F1 Score Results:
  Macro-F1: 0.8465

  Per-class F1 scores:
    Class 0: 0.8542
    Class 1: 0.8553
    Class 2: 0.8110
    Class 3: 0.8704
    Class 4: 0.8418

  best model! save...


## During Testing

When you run test.py, you'll see F1 scores at the end:

Example output:
--------------

Test phase:
  [████████████████████████████████] 100% | Acc 84.67%

  📊 F1 Score Results:
  Macro-F1: 0.8465

  Per-class F1 scores:
    Class 0: 0.8542
    Class 1: 0.8553
    Class 2: 0.8110
    Class 3: 0.8704
    Class 4: 0.8418

  600 Test Acc = 84.67% +- 2.15%


## What the F1 Scores Tell You

1. **Macro-F1**: Average F1 score across all classes
   - Ranges from 0 to 1 (higher is better)
   - Gives equal weight to all classes
   - Good for understanding overall balanced performance

2. **Per-class F1**: Individual F1 score for each class
   - Shows which classes the model performs well/poorly on
   - Useful for identifying imbalanced learning
   - Helps guide improvements for specific classes

3. **When F1 differs from Accuracy**:
   - If Macro-F1 is much lower than accuracy, some classes may be underperforming
   - Look at per-class F1 scores to identify problematic classes
   - Consider data augmentation or class weighting for low-F1 classes


## Using the Information

### In WandB Dashboard
F1 scores are automatically logged to WandB (if enabled):
- 'Val Acc': Validation accuracy
- 'Val Macro-F1': Macro-averaged F1 score

### In Your Experiments
Use F1 scores to:
- Compare model variants (which has better balanced performance?)
- Identify weak classes (which classes need more training data?)
- Track improvement over epochs (is macro-F1 increasing?)
- Evaluate fairness (are all classes performing similarly?)


## Example Scenarios

### Scenario 1: High accuracy, low F1
Acc: 90%, Macro-F1: 0.65
→ Model may be biased toward majority classes
→ Check per-class F1 to find underperforming classes

### Scenario 2: Balanced performance
Acc: 85%, Macro-F1: 0.84
→ Model performs well across all classes
→ Per-class F1 scores are similar

### Scenario 3: One weak class
Class 0: 0.92, Class 1: 0.89, Class 2: 0.45, Class 3: 0.88, Class 4: 0.91
→ Class 2 needs attention
→ Consider adding more training data or augmentation for Class 2
""")

print("=" * 70)
print("\nFor more details, see F1_SCORES_IMPLEMENTATION.md")
print("=" * 70)
