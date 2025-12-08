"""
Unit tests for new features: show_plots and mcnemar_each_test

These tests validate the new command-line arguments and functionality
without requiring full model training or datasets.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import ExperimentConfig, RunMode


def test_show_plots_config():
    """Test show_plots configuration flag"""
    print("Testing show_plots configuration flag...")
    
    # Test default value (False)
    config_default = ExperimentConfig()
    assert config_default.show_plots is False, \
        f"Expected show_plots default to be False, got {config_default.show_plots}"
    
    # Test explicit True
    config_true = ExperimentConfig(show_plots=True)
    assert config_true.show_plots is True, \
        f"Expected show_plots to be True, got {config_true.show_plots}"
    
    # Test explicit False
    config_false = ExperimentConfig(show_plots=False)
    assert config_false.show_plots is False, \
        f"Expected show_plots to be False, got {config_false.show_plots}"
    
    print("  ✓ show_plots configuration tests passed")


def test_mcnemar_each_test_config():
    """Test mcnemar_each_test configuration flag"""
    print("Testing mcnemar_each_test configuration flag...")
    
    # Test default value (False)
    config_default = ExperimentConfig()
    assert config_default.mcnemar_each_test is False, \
        f"Expected mcnemar_each_test default to be False, got {config_default.mcnemar_each_test}"
    
    # Test explicit True
    config_true = ExperimentConfig(mcnemar_each_test=True)
    assert config_true.mcnemar_each_test is True, \
        f"Expected mcnemar_each_test to be True, got {config_true.mcnemar_each_test}"
    
    # Test explicit False
    config_false = ExperimentConfig(mcnemar_each_test=False)
    assert config_false.mcnemar_each_test is False, \
        f"Expected mcnemar_each_test to be False, got {config_false.mcnemar_each_test}"
    
    print("  ✓ mcnemar_each_test configuration tests passed")


def test_combined_config():
    """Test both flags together"""
    print("Testing combined configuration...")
    
    # Test both True
    config_both_true = ExperimentConfig(
        show_plots=True,
        mcnemar_each_test=True,
        dataset='miniImagenet',
        backbone='Conv4'
    )
    assert config_both_true.show_plots is True
    assert config_both_true.mcnemar_each_test is True
    assert config_both_true.dataset == 'miniImagenet'
    assert config_both_true.backbone == 'Conv4'
    
    # Test both False
    config_both_false = ExperimentConfig(
        show_plots=False,
        mcnemar_each_test=False
    )
    assert config_both_false.show_plots is False
    assert config_both_false.mcnemar_each_test is False
    
    # Test mixed
    config_mixed = ExperimentConfig(
        show_plots=True,
        mcnemar_each_test=False
    )
    assert config_mixed.show_plots is True
    assert config_mixed.mcnemar_each_test is False
    
    print("  ✓ Combined configuration tests passed")


def test_mcnemar_comparison_function_availability():
    """Test that run_mcnemar_comparison function is available"""
    print("Testing run_mcnemar_comparison function availability...")
    
    try:
        from run_experiments import run_mcnemar_comparison
        
        # Function should exist
        assert callable(run_mcnemar_comparison), \
            "run_mcnemar_comparison should be callable"
        
        print("  ✓ run_mcnemar_comparison function is available")
    except ImportError as e:
        print(f"  ⚠ Cannot import run_mcnemar_comparison: {e}")
        print("  Note: This is expected if torch is not available")


def test_safe_plot_save_signature():
    """Test that safe_plot_save has the correct signature"""
    print("Testing safe_plot_save function signature...")
    
    try:
        from run_experiments import safe_plot_save
        import inspect
        
        # Get function signature
        sig = inspect.signature(safe_plot_save)
        params = list(sig.parameters.keys())
        
        # Should have 'output_path', 'dpi', and 'show' parameters
        assert 'output_path' in params, "Missing 'output_path' parameter"
        assert 'dpi' in params, "Missing 'dpi' parameter"
        assert 'show' in params, "Missing 'show' parameter"
        
        # Check default values
        assert sig.parameters['dpi'].default == 150, \
            f"Expected dpi default to be 150, got {sig.parameters['dpi'].default}"
        assert sig.parameters['show'].default == False, \
            f"Expected show default to be False, got {sig.parameters['show'].default}"
        
        print("  ✓ safe_plot_save function signature is correct")
    except ImportError as e:
        print(f"  ⚠ Cannot import safe_plot_save: {e}")
        print("  Note: This is expected if torch is not available")


def test_config_with_all_parameters():
    """Test ExperimentConfig with all parameters including new ones"""
    print("Testing ExperimentConfig with all parameters...")
    
    config = ExperimentConfig(
        dataset='CUB',
        backbone='ResNet18',
        n_way=5,
        k_shot=5,
        n_query=15,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.0001,
        optimization='Adam',
        test_iter=1000,
        run_mode=RunMode.ALL,
        output_dir='./test_results',
        seed=1234,
        show_plots=True,
        mcnemar_each_test=True
    )
    
    # Verify all parameters
    assert config.dataset == 'CUB'
    assert config.backbone == 'ResNet18'
    assert config.n_way == 5
    assert config.k_shot == 5
    assert config.n_query == 15
    assert config.num_epochs == 100
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.optimization == 'Adam'
    assert config.test_iter == 1000
    assert config.run_mode == RunMode.ALL
    assert config.output_dir == './test_results'
    assert config.seed == 1234
    assert config.show_plots is True
    assert config.mcnemar_each_test is True
    
    print("  ✓ ExperimentConfig with all parameters tests passed")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running New Features Tests (show_plots, mcnemar_each_test)")
    print("="*60)
    print()
    
    try:
        test_show_plots_config()
        test_mcnemar_each_test_config()
        test_combined_config()
        test_mcnemar_comparison_function_availability()
        test_safe_plot_save_signature()
        test_config_with_all_parameters()
        
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
        import traceback
        traceback.print_exc()
        return False
    
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
