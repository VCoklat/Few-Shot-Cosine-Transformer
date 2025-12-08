"""
Unit tests for the unified experiment runner configuration classes.

These tests validate the configuration system without requiring
external dependencies like torch or datasets.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    ExperimentConfig,
    AblationExperimentConfig,
    VICComponents,
    RunMode,
    ABLATION_EXPERIMENTS
)


def test_vic_components():
    """Test VICComponents class"""
    print("Testing VICComponents...")
    
    # Test full VIC
    vic_full = VICComponents(
        invariance=True,
        covariance=True,
        variance=True,
        dynamic_weight=True
    )
    assert str(vic_full) == "ICVD", f"Expected 'ICVD', got '{str(vic_full)}'"
    
    # Test baseline (no components)
    vic_baseline = VICComponents(
        invariance=False,
        covariance=False,
        variance=False,
        dynamic_weight=False
    )
    assert str(vic_baseline) == "Baseline", f"Expected 'Baseline', got '{str(vic_baseline)}'"
    
    # Test partial components
    vic_partial = VICComponents(
        invariance=True,
        covariance=False,
        variance=True,
        dynamic_weight=False
    )
    assert str(vic_partial) == "IV", f"Expected 'IV', got '{str(vic_partial)}'"
    
    # Test to_dict
    vic_dict = vic_full.to_dict()
    assert vic_dict['invariance'] == True
    assert vic_dict['covariance'] == True
    assert vic_dict['variance'] == True
    assert vic_dict['dynamic_weight'] == True
    
    print("  ✓ VICComponents tests passed")


def test_ablation_experiment_config():
    """Test AblationExperimentConfig class"""
    print("Testing AblationExperimentConfig...")
    
    vic = VICComponents(invariance=True, covariance=True, variance=True, dynamic_weight=True)
    config = AblationExperimentConfig(
        name="Test_Experiment",
        vic_components=vic,
        description="Test description"
    )
    
    assert config.name == "Test_Experiment"
    assert config.description == "Test description"
    
    # Test to_dict
    config_dict = config.to_dict()
    assert config_dict['name'] == "Test_Experiment"
    assert config_dict['description'] == "Test description"
    assert 'vic_components' in config_dict
    
    print("  ✓ AblationExperimentConfig tests passed")


def test_ablation_experiments():
    """Test pre-defined ablation experiments"""
    print("Testing pre-defined ABLATION_EXPERIMENTS...")
    
    # Check all 8 experiments exist
    expected_keys = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
    for key in expected_keys:
        assert key in ABLATION_EXPERIMENTS, f"Missing experiment: {key}"
    
    # Test E1 (Full model)
    e1 = ABLATION_EXPERIMENTS['E1']
    assert e1.name == 'E1_Full'
    assert e1.vic_components.invariance == True
    assert e1.vic_components.covariance == True
    assert e1.vic_components.variance == True
    assert e1.vic_components.dynamic_weight == True
    
    # Test E6 (Baseline)
    e6 = ABLATION_EXPERIMENTS['E6']
    assert e6.name == 'E6_Baseline'
    assert e6.vic_components.invariance == False
    assert e6.vic_components.covariance == False
    assert e6.vic_components.variance == False
    assert e6.vic_components.dynamic_weight == False
    
    # Test E2 (Invariance only + dynamic)
    e2 = ABLATION_EXPERIMENTS['E2']
    assert e2.name == 'E2_InvDyn'
    assert e2.vic_components.invariance == True
    assert e2.vic_components.covariance == False
    assert e2.vic_components.variance == False
    assert e2.vic_components.dynamic_weight == True
    
    print("  ✓ All 8 pre-defined ablation experiments validated")


def test_run_mode():
    """Test RunMode enum"""
    print("Testing RunMode...")
    
    assert RunMode.ALL.value == "all"
    assert RunMode.TRAIN_TEST.value == "train_test"
    assert RunMode.ABLATION.value == "ablation"
    assert RunMode.QUALITATIVE.value == "qualitative"
    assert RunMode.FEATURE_ANALYSIS.value == "feature_analysis"
    assert RunMode.MCNEMAR.value == "mcnemar"
    
    # Test enum creation from string
    mode = RunMode("ablation")
    assert mode == RunMode.ABLATION
    
    print("  ✓ RunMode tests passed")


def test_experiment_config():
    """Test ExperimentConfig class"""
    print("Testing ExperimentConfig...")
    
    config = ExperimentConfig(
        dataset='miniImagenet',
        backbone='Conv4',
        n_way=5,
        k_shot=1,
        n_query=16,
        num_epochs=50,
        test_iter=600,
        run_mode=RunMode.ALL,
        output_dir='./results',
        seed=4040
    )
    
    # Test basic attributes
    assert config.dataset == 'miniImagenet'
    assert config.backbone == 'Conv4'
    assert config.n_way == 5
    assert config.k_shot == 1
    
    # Test experiment name generation
    exp_name = config.get_experiment_name()
    assert exp_name == 'miniImagenet_Conv4_5w1s', f"Expected 'miniImagenet_Conv4_5w1s', got '{exp_name}'"
    
    # Test output paths generation
    paths = config.get_output_paths()
    assert 'base' in paths
    assert 'quantitative' in paths
    assert 'qualitative' in paths
    assert 'ablation' in paths
    assert 'mcnemar' in paths
    assert 'feature_analysis' in paths
    
    expected_base = './results/miniImagenet_Conv4_5w1s'
    assert paths['base'] == expected_base, f"Expected '{expected_base}', got '{paths['base']}'"
    
    # Test to_dict
    config_dict = config.to_dict()
    assert config_dict['dataset'] == 'miniImagenet'
    assert config_dict['backbone'] == 'Conv4'
    assert config_dict['n_way'] == 5
    assert config_dict['k_shot'] == 1
    assert config_dict['run_mode'] == 'all'
    
    print("  ✓ ExperimentConfig tests passed")


def test_default_ablation_experiments():
    """Test that default ablation experiments list is correct"""
    print("Testing default ablation experiments list...")
    
    config = ExperimentConfig()
    default_experiments = config.ablation_experiments
    
    expected_keys = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
    assert default_experiments == expected_keys, \
        f"Expected {expected_keys}, got {default_experiments}"
    
    print("  ✓ Default ablation experiments validated")


def test_vic_components_combinations():
    """Test various VIC component combinations match ablation experiments"""
    print("Testing VIC component combinations...")
    
    # E1: Full (ICVD)
    e1_vic = ABLATION_EXPERIMENTS['E1'].vic_components
    assert str(e1_vic) == "ICVD"
    
    # E2: Invariance + Dynamic (ID)
    e2_vic = ABLATION_EXPERIMENTS['E2'].vic_components
    assert str(e2_vic) == "ID"
    
    # E3: Invariance + Covariance + Dynamic (ICD)
    e3_vic = ABLATION_EXPERIMENTS['E3'].vic_components
    assert str(e3_vic) == "ICD"
    
    # E4: Invariance + Variance + Dynamic (IVD)
    e4_vic = ABLATION_EXPERIMENTS['E4'].vic_components
    assert str(e4_vic) == "IVD"
    
    # E5: Full without Dynamic (ICV)
    e5_vic = ABLATION_EXPERIMENTS['E5'].vic_components
    assert str(e5_vic) == "ICV"
    
    # E6: Baseline (empty)
    e6_vic = ABLATION_EXPERIMENTS['E6'].vic_components
    assert str(e6_vic) == "Baseline"
    
    # E7: Covariance + Dynamic (CD)
    e7_vic = ABLATION_EXPERIMENTS['E7'].vic_components
    assert str(e7_vic) == "CD"
    
    # E8: Variance + Dynamic (VD)
    e8_vic = ABLATION_EXPERIMENTS['E8'].vic_components
    assert str(e8_vic) == "VD"
    
    print("  ✓ All VIC component combinations validated")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Unified Experiment Runner Configuration Tests")
    print("="*60)
    print()
    
    try:
        test_vic_components()
        test_ablation_experiment_config()
        test_ablation_experiments()
        test_run_mode()
        test_experiment_config()
        test_default_ablation_experiments()
        test_vic_components_combinations()
        
        print()
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return True
    
    except AssertionError as e:
        print()
        print("="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        return False
    
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("="*60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
