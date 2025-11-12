# F1 Scores Implementation

## Overview
This document describes the implementation of F1 scores for all classes during validation and test phases in the Few-Shot Cosine Transformer project.

## Problem Statement
The user requested F1 scores for all classes to be displayed during validation and test phases to better evaluate model performance across different classes.

## Implementation

### Test Phase
✅ **Already Implemented** - The `test.py` file already calculates and displays F1 scores:
- Per-class F1 scores for all classes
- Macro-averaged F1 score
- Located in `direct_test()` function (lines 80-90)

### Validation Phase
✅ **Now Implemented** - Updated `methods/meta_template.py`:
- Modified `val_loop()` method to calculate F1 scores
- Per-class F1 scores for all classes
- Macro-averaged F1 score
- F1 scores are displayed after validation accuracy
- Macro-F1 is logged to WandB when enabled

## Changes Made

### File: `methods/meta_template.py`
```python
def val_loop(self, val_loader, epoch, wandb_flag, record = None):
    from sklearn.metrics import f1_score
    
    # ... existing code ...
    
    # Collect predictions and labels
    all_preds = []
    all_labels = []
    
    for i, (x,_) in enumerate(val_loader):
        # Get predictions for F1 score calculation
        scores = self.set_forward(x)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.n_way), self.n_query)
        
        all_preds.extend(pred.tolist())
        all_labels.extend(y.tolist())
        
        # ... existing accuracy calculation ...
    
    # Calculate and display per-class F1 scores
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print('Val Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    print(f"\n📊 Validation F1 Score Results:")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(class_f1):
        print(f"  Class {i}: {f1:.4f}")
    
    if wandb_flag:
        wandb.log({'Val Acc': acc_mean, 'Val Macro-F1': macro_f1}, step=epoch + 1)
```

## Output Example

### During Validation
```
Val Acc = 85.32% +- 1.24%

📊 Validation F1 Score Results:
Macro-F1: 0.8465

Per-class F1 scores:
  Class 0: 0.8542
  Class 1: 0.8553
  Class 2: 0.8110
  Class 3: 0.8704
  Class 4: 0.8418
```

### During Test
```
Test       | Acc 84.67%

📊 F1 Score Results:
Macro-F1: 0.8465

Per-class F1 scores:
  Class 0: 0.8542
  Class 1: 0.8553
  Class 2: 0.8110
  Class 3: 0.8704
  Class 4: 0.8418

600 Test Acc = 84.67% +- 2.15%
```

## Testing

A comprehensive test file `test_f1_validation.py` has been created to verify:
1. F1 score calculation logic works correctly
2. `val_loop` method has been properly updated with F1 calculation
3. All F1 scores are valid (between 0 and 1)
4. Per-class and macro-averaged F1 scores are computed

Run the test:
```bash
python test_f1_validation.py
```

## Dependencies

The implementation uses `scikit-learn` for F1 score calculation, which is already included in `requirements.txt`.

## Benefits

1. **Better Class-level Insights**: Per-class F1 scores reveal performance differences across classes
2. **Balanced Performance Metric**: Macro-F1 provides a balanced view across all classes
3. **Consistency**: Validation and test phases now report the same metrics
4. **WandB Integration**: F1 scores are automatically logged to WandB for experiment tracking
5. **No Breaking Changes**: All existing functionality remains intact

## Notes

- F1 scores are particularly useful in imbalanced classification scenarios
- The implementation follows the same pattern used in `test.py` for consistency
- Changes are minimal and focused on the validation loop only
- Test phase already had F1 scores, this update makes validation consistent
