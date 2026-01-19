import torch
import torch.nn as nn
from methods.optimal_few_shot import OptimalFewShotModel
from methods.baselines import ProtoNet, MatchingNet, MetaBaseline, RelationNet

def verify_optimal_model():
    print("Verifying OptimalFewShotModel...")
    # Initialize model
    model = OptimalFewShotModel(
        model_func=None, # Use default Conv4
        n_way=5, k_shot=1, n_query=15,
        feature_dim=64,
        dataset='miniImagenet'
    )
    
    # Enable new flags
    model.vic.use_projector = True
    model.explicit_invariance = True # Sanity check enabled
    model = model.cuda()
    
    # Dummy input [n_way * (k+q), C, H, W]
    # 5 way, 1 shot, 15 query = 5 * 16 = 80 samples
    x = torch.randn(80, 3, 84, 84).cuda()
    
    # Forward loss
    try:
        acc, loss = model.set_forward_loss(x)
        print(f"  Forward pass successful. Acc: {acc:.2f}, Loss: {loss.item():.4f}")
        
        # Check if loss includes invariance
        # We can't easily check internal loss components without modifying return, 
        # but if it runs without error, augmentation logic is likely valid.
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

def verify_baselines():
    print("\nVerifying Baselines...")
    models = [ProtoNet, MatchingNet, MetaBaseline, RelationNet]
    
    for ModelClass in models:
        print(f"  Checking {ModelClass.__name__}...")
        try:
            # Mock feature model function
            feature_model = lambda: nn.Sequential(nn.Conv2d(3, 64, 3), nn.Flatten(), nn.Linear(53824, 64))
            
            model = ModelClass(feature_model, n_way=5, k_shot=1, n_query=15).cuda()
            x = torch.randn(80, 3, 84, 84).cuda()
            
            acc, loss = model.set_forward_loss(x)
            print(f"    Pass successful. Acc: {acc:.2f}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_optimal_model()
    verify_baselines()
