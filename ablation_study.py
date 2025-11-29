"""
Ablation Study and McNemar's Test Module for Few-Shot Classification

This module provides:
1. Ablation study framework for analyzing component contributions
2. McNemar's Test for statistical comparison of classification algorithms
3. Utilities for comparing predictions between different model configurations

McNemar's Test is used for comparing two algorithms on a classification task,
focusing on the differences in their error rates (comparing instances one 
algorithm got right and the other got wrong).

Author: dvh
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class AblationType(Enum):
    """Types of ablation configurations"""
    FULL_MODEL = "full_model"
    NO_SE_BLOCKS = "no_se_blocks"
    NO_COSINE_ATTENTION = "no_cosine_attention"
    NO_VIC_REGULARIZATION = "no_vic_regularization"
    NO_DYNAMIC_WEIGHTING = "no_dynamic_weighting"
    SINGLE_ATTENTION_HEAD = "single_attention_head"
    REDUCED_HEADS = "reduced_heads"
    CUSTOM = "custom"


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment"""
    name: str
    ablation_type: AblationType
    description: str
    method: str = "FSCT_cosine"
    variant: str = "cosine"
    heads: int = 8
    use_se: bool = True
    use_vic: bool = True
    use_dynamic_weights: bool = True
    custom_params: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'ablation_type': self.ablation_type.value,
            'description': self.description,
            'method': self.method,
            'variant': self.variant,
            'heads': self.heads,
            'use_se': self.use_se,
            'use_vic': self.use_vic,
            'use_dynamic_weights': self.use_dynamic_weights,
            'custom_params': self.custom_params
        }


@dataclass
class AblationResult:
    """Results from an ablation experiment"""
    config: AblationConfig
    accuracy: float
    std: float
    predictions: np.ndarray
    true_labels: np.ndarray
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    additional_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary (excluding numpy arrays)"""
        return {
            'config': self.config.to_dict(),
            'accuracy': self.accuracy,
            'std': self.std,
            'confidence_interval': list(self.confidence_interval),
            'additional_metrics': self.additional_metrics
        }


# ========================================================================
# McNemar's Test Implementation
# ========================================================================

def compute_contingency_table(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    true_labels: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute the contingency table for McNemar's test.
    
    The contingency table counts:
    - n00: Both algorithms wrong
    - n01: Algorithm A wrong, Algorithm B correct
    - n10: Algorithm A correct, Algorithm B wrong  
    - n11: Both algorithms correct
    
    Args:
        predictions_a: Predictions from algorithm A
        predictions_b: Predictions from algorithm B
        true_labels: True labels
    
    Returns:
        Tuple of (n00, n01, n10, n11)
    """
    if len(predictions_a) != len(predictions_b) or len(predictions_a) != len(true_labels):
        raise ValueError("All arrays must have the same length")
    
    correct_a = (predictions_a == true_labels)
    correct_b = (predictions_b == true_labels)
    
    n00 = np.sum(~correct_a & ~correct_b)  # Both wrong
    n01 = np.sum(~correct_a & correct_b)   # A wrong, B correct
    n10 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    n11 = np.sum(correct_a & correct_b)    # Both correct
    
    return int(n00), int(n01), int(n10), int(n11)


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    true_labels: np.ndarray,
    correction: bool = True,
    exact: bool = False
) -> Dict:
    """
    Perform McNemar's test to compare two classification algorithms.
    
    McNemar's test is used to determine if there is a statistically significant 
    difference between the error rates of two classifiers on the same test set.
    
    The null hypothesis is that the two algorithms have the same error rate.
    
    Args:
        predictions_a: Predictions from algorithm A
        predictions_b: Predictions from algorithm B  
        true_labels: True class labels
        correction: If True, apply continuity correction (Edwards' correction)
        exact: If True, use exact binomial test instead of chi-squared approximation
               (recommended when n01 + n10 < 25)
    
    Returns:
        Dictionary containing:
            - contingency_table: (n00, n01, n10, n11)
            - statistic: Test statistic
            - p_value: P-value of the test
            - significant_at_0.05: Boolean indicating significance at alpha=0.05
            - significant_at_0.01: Boolean indicating significance at alpha=0.01
            - algorithm_a_better: True if A performs better than B
            - algorithm_b_better: True if B performs better than A
            - discordant_pairs: Number of instances where algorithms disagree
            - effect_description: Human-readable description of the result
    """
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    true_labels = np.asarray(true_labels)
    
    n00, n01, n10, n11 = compute_contingency_table(predictions_a, predictions_b, true_labels)
    
    # Number of discordant pairs (where algorithms disagree on correctness)
    discordant_pairs = n01 + n10
    
    # Handle edge case where there are no discordant pairs
    if discordant_pairs == 0:
        return {
            'contingency_table': (n00, n01, n10, n11),
            'statistic': 0.0,
            'p_value': 1.0,
            'significant_at_0.05': False,
            'significant_at_0.01': False,
            'algorithm_a_better': False,
            'algorithm_b_better': False,
            'discordant_pairs': 0,
            'effect_description': "No difference: Both algorithms agree on all instances",
            'test_type': 'none'
        }
    
    # Determine which test to use
    if exact or discordant_pairs < 25:
        # Use exact binomial test
        # Under null hypothesis, n01 ~ Binomial(n01+n10, 0.5)
        # Use binomtest (new API) if available, otherwise fall back to binom_test
        try:
            result = stats.binomtest(n01, n=discordant_pairs, p=0.5, alternative='two-sided')
            p_value = result.pvalue
        except AttributeError:
            # Fall back to deprecated binom_test for older scipy versions
            p_value = stats.binom_test(n01, n=discordant_pairs, p=0.5, alternative='two-sided')
        statistic = n01  # The test statistic is just the count
        test_type = 'exact_binomial'
    else:
        # Use chi-squared approximation
        if correction:
            # Edwards' continuity correction
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        else:
            statistic = (n01 - n10) ** 2 / (n01 + n10)
        
        # Chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        test_type = 'chi_squared' + ('_corrected' if correction else '')
    
    # Determine which algorithm is better
    algorithm_a_better = n10 > n01  # A correct when B wrong, more than vice versa
    algorithm_b_better = n01 > n10
    
    # Generate effect description
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    if algorithm_a_better and p_value < 0.05:
        effect_description = f"Algorithm A significantly outperforms B ({significance})"
    elif algorithm_b_better and p_value < 0.05:
        effect_description = f"Algorithm B significantly outperforms A ({significance})"
    else:
        effect_description = f"No significant difference between algorithms ({significance})"
    
    return {
        'contingency_table': (n00, n01, n10, n11),
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'algorithm_a_better': algorithm_a_better and p_value < 0.05,
        'algorithm_b_better': algorithm_b_better and p_value < 0.05,
        'discordant_pairs': discordant_pairs,
        'effect_description': effect_description,
        'test_type': test_type,
        'n01': n01,  # A wrong, B correct
        'n10': n10   # A correct, B wrong
    }


def mcnemar_test_multiple(
    predictions_list: List[np.ndarray],
    algorithm_names: List[str],
    true_labels: np.ndarray,
    correction: bool = True
) -> Dict:
    """
    Perform pairwise McNemar's tests for multiple algorithms.
    
    Args:
        predictions_list: List of prediction arrays from each algorithm
        algorithm_names: Names of the algorithms
        true_labels: True class labels
        correction: If True, apply continuity correction
    
    Returns:
        Dictionary with pairwise comparison results and summary
    """
    n_algorithms = len(predictions_list)
    if len(algorithm_names) != n_algorithms:
        raise ValueError("Number of names must match number of prediction arrays")
    
    results = {
        'algorithm_names': algorithm_names,
        'pairwise_comparisons': [],
        'summary_matrix': np.zeros((n_algorithms, n_algorithms)),
        'significant_differences': []
    }
    
    # Perform all pairwise comparisons
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            comparison = mcnemar_test(
                predictions_list[i],
                predictions_list[j],
                true_labels,
                correction=correction
            )
            comparison['algorithm_a'] = algorithm_names[i]
            comparison['algorithm_b'] = algorithm_names[j]
            results['pairwise_comparisons'].append(comparison)
            
            # Update summary matrix with p-values
            results['summary_matrix'][i, j] = comparison['p_value']
            results['summary_matrix'][j, i] = comparison['p_value']
            
            # Track significant differences
            if comparison['significant_at_0.05']:
                winner = algorithm_names[i] if comparison['algorithm_a_better'] else algorithm_names[j]
                loser = algorithm_names[j] if comparison['algorithm_a_better'] else algorithm_names[i]
                results['significant_differences'].append({
                    'winner': winner,
                    'loser': loser,
                    'p_value': comparison['p_value']
                })
    
    # Convert numpy array to list for JSON serialization
    results['summary_matrix'] = results['summary_matrix'].tolist()
    
    return results


# ========================================================================
# Ablation Study Framework
# ========================================================================

class AblationStudy:
    """
    Framework for conducting ablation studies on few-shot learning models.
    
    Supports:
    - Running multiple ablation configurations
    - Statistical comparison using McNemar's test
    - Result visualization and reporting
    """
    
    def __init__(self, baseline_name: str = "Full Model"):
        """
        Initialize ablation study.
        
        Args:
            baseline_name: Name of the baseline configuration
        """
        self.baseline_name = baseline_name
        self.results: Dict[str, AblationResult] = {}
        self.baseline_result: Optional[AblationResult] = None
    
    @staticmethod
    def get_standard_ablations() -> List[AblationConfig]:
        """
        Get standard ablation configurations for the Few-Shot Cosine Transformer.
        
        Returns:
            List of AblationConfig objects for standard ablations
        """
        return [
            AblationConfig(
                name="Full Model (Baseline)",
                ablation_type=AblationType.FULL_MODEL,
                description="Complete model with all components",
                method="FSCT_cosine",
                variant="cosine",
                heads=8,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="Without Cosine Attention",
                ablation_type=AblationType.NO_COSINE_ATTENTION,
                description="Using softmax attention instead of cosine attention",
                method="FSCT_softmax",
                variant="softmax",
                heads=8,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="Single Attention Head",
                ablation_type=AblationType.SINGLE_ATTENTION_HEAD,
                description="Using only 1 attention head instead of 8",
                method="FSCT_cosine",
                variant="cosine",
                heads=1,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="Reduced Heads (4)",
                ablation_type=AblationType.REDUCED_HEADS,
                description="Using 4 attention heads instead of 8",
                method="FSCT_cosine",
                variant="cosine",
                heads=4,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="No SE Blocks",
                ablation_type=AblationType.NO_SE_BLOCKS,
                description="Model without Squeeze-and-Excitation blocks in backbone",
                method="FSCT_cosine",
                variant="cosine",
                heads=8,
                use_se=False,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="No Dynamic Weighting",
                ablation_type=AblationType.NO_DYNAMIC_WEIGHTING,
                description="Using uniform weights for prototype computation",
                method="FSCT_cosine",
                variant="cosine",
                heads=8,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=False
            ),
            AblationConfig(
                name="No VIC Regularization",
                ablation_type=AblationType.NO_VIC_REGULARIZATION,
                description="Model without VIC regularization loss",
                method="FSCT_cosine",
                variant="cosine",
                heads=8,
                use_se=True,
                use_vic=False,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="CTX Softmax Baseline",
                ablation_type=AblationType.CUSTOM,
                description="CTX method with softmax attention",
                method="CTX_softmax",
                variant="softmax",
                heads=8,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
            AblationConfig(
                name="CTX Cosine",
                ablation_type=AblationType.CUSTOM,
                description="CTX method with cosine attention",
                method="CTX_cosine",
                variant="cosine",
                heads=8,
                use_se=True,
                use_vic=True,
                use_dynamic_weights=True
            ),
        ]
    
    def add_result(
        self,
        name: str,
        config: AblationConfig,
        accuracy: float,
        std: float,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        confidence_interval: Tuple[float, float] = (0.0, 0.0),
        additional_metrics: Dict = None,
        is_baseline: bool = False
    ):
        """
        Add an ablation result.
        
        Args:
            name: Name of the configuration
            config: AblationConfig object
            accuracy: Mean accuracy
            std: Standard deviation
            predictions: Model predictions
            true_labels: True labels
            confidence_interval: 95% confidence interval
            additional_metrics: Additional metrics dictionary
            is_baseline: If True, set as baseline for comparisons
        """
        result = AblationResult(
            config=config,
            accuracy=accuracy,
            std=std,
            predictions=predictions,
            true_labels=true_labels,
            confidence_interval=confidence_interval,
            additional_metrics=additional_metrics or {}
        )
        
        self.results[name] = result
        
        if is_baseline:
            self.baseline_result = result
            self.baseline_name = name
    
    def compare_to_baseline(self, name: str) -> Optional[Dict]:
        """
        Compare a configuration to the baseline using McNemar's test.
        
        Args:
            name: Name of the configuration to compare
        
        Returns:
            McNemar's test results or None if comparison not possible
        """
        if self.baseline_result is None:
            warnings.warn("No baseline set. Use add_result with is_baseline=True")
            return None
        
        if name not in self.results:
            warnings.warn(f"Configuration '{name}' not found")
            return None
        
        result = self.results[name]
        
        comparison = mcnemar_test(
            self.baseline_result.predictions,
            result.predictions,
            self.baseline_result.true_labels
        )
        
        comparison['baseline_name'] = self.baseline_name
        comparison['compared_name'] = name
        comparison['baseline_accuracy'] = self.baseline_result.accuracy
        comparison['compared_accuracy'] = result.accuracy
        comparison['accuracy_difference'] = result.accuracy - self.baseline_result.accuracy
        
        return comparison
    
    def compare_all(self) -> Dict:
        """
        Compare all configurations using pairwise McNemar's tests.
        
        Returns:
            Dictionary with all pairwise comparisons
        """
        if len(self.results) < 2:
            warnings.warn("Need at least 2 results for comparison")
            return {}
        
        names = list(self.results.keys())
        predictions_list = [self.results[name].predictions for name in names]
        
        # Use the true labels from the first result (should be same for all)
        true_labels = list(self.results.values())[0].true_labels
        
        return mcnemar_test_multiple(predictions_list, names, true_labels)
    
    def compute_ablation_impact(self) -> List[Dict]:
        """
        Compute the impact of each ablation relative to baseline.
        
        Returns:
            List of dictionaries with impact metrics for each ablation
        """
        if self.baseline_result is None:
            warnings.warn("No baseline set")
            return []
        
        impacts = []
        baseline_acc = self.baseline_result.accuracy
        
        for name, result in self.results.items():
            if name == self.baseline_name:
                continue
            
            acc_diff = result.accuracy - baseline_acc
            relative_drop = (acc_diff / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            # Perform McNemar's test
            comparison = self.compare_to_baseline(name)
            
            impact = {
                'name': name,
                'config': result.config.to_dict(),
                'accuracy': result.accuracy,
                'baseline_accuracy': baseline_acc,
                'absolute_difference': acc_diff,
                'relative_change_pct': relative_drop,
                'is_significant': comparison['significant_at_0.05'] if comparison else False,
                'p_value': comparison['p_value'] if comparison else 1.0,
                'mcnemar_result': comparison
            }
            impacts.append(impact)
        
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x['absolute_difference']), reverse=True)
        
        return impacts
    
    def generate_report(self) -> str:
        """
        Generate a formatted text report of ablation study results.
        
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 80)
        
        if self.baseline_result:
            lines.append(f"\nBaseline: {self.baseline_name}")
            lines.append(f"Baseline Accuracy: {self.baseline_result.accuracy:.4f} "
                        f"(±{self.baseline_result.std:.4f})")
            if self.baseline_result.confidence_interval != (0.0, 0.0):
                lines.append(f"95% CI: [{self.baseline_result.confidence_interval[0]:.4f}, "
                            f"{self.baseline_result.confidence_interval[1]:.4f}]")
        
        lines.append("\n" + "-" * 80)
        lines.append("ABLATION RESULTS")
        lines.append("-" * 80)
        
        impacts = self.compute_ablation_impact()
        
        lines.append(f"\n{'Configuration':<40} {'Accuracy':>10} {'Δ Acc':>10} {'Δ %':>10} {'p-value':>10} {'Sig.':>6}")
        lines.append("-" * 86)
        
        for impact in impacts:
            sig_marker = "***" if impact.get('is_significant', False) else ""
            lines.append(
                f"{impact['name']:<40} "
                f"{impact['accuracy']:>10.4f} "
                f"{impact['absolute_difference']:>+10.4f} "
                f"{impact['relative_change_pct']:>+10.2f}% "
                f"{impact['p_value']:>10.4f} "
                f"{sig_marker:>6}"
            )
        
        lines.append("\n*** = Statistically significant difference (p < 0.05)")
        
        # McNemar's test details for significant results
        sig_impacts = [i for i in impacts if i.get('is_significant', False)]
        if sig_impacts:
            lines.append("\n" + "-" * 80)
            lines.append("SIGNIFICANT DIFFERENCES (McNemar's Test)")
            lines.append("-" * 80)
            
            for impact in sig_impacts:
                mcnemar = impact.get('mcnemar_result', {})
                lines.append(f"\n{impact['name']}:")
                lines.append(f"  {mcnemar.get('effect_description', 'N/A')}")
                lines.append(f"  Discordant pairs: {mcnemar.get('discordant_pairs', 'N/A')}")
                lines.append(f"  Contingency table: {mcnemar.get('contingency_table', 'N/A')}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str):
        """
        Save ablation study results to JSON file.
        
        Args:
            filepath: Path to output file
        """
        output = {
            'baseline_name': self.baseline_name,
            'baseline_result': self.baseline_result.to_dict() if self.baseline_result else None,
            'ablation_results': {name: result.to_dict() for name, result in self.results.items()},
            'ablation_impacts': self.compute_ablation_impact(),
            'pairwise_comparisons': self.compare_all()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    @classmethod
    def load_results(cls, filepath: str) -> 'AblationStudy':
        """
        Load ablation study results from JSON file.
        
        Args:
            filepath: Path to input file
        
        Returns:
            AblationStudy object with loaded results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        study = cls(baseline_name=data.get('baseline_name', 'Baseline'))
        # Note: This loads summary data but not the actual predictions
        # For full functionality, results need to be regenerated
        
        return study
    
    def run_ablation_experiment(
        self,
        config: AblationConfig,
        loader,
        model_func,
        n_way: int,
        k_shot: int,
        n_query: int,
        dataset: str = 'miniImagenet',
        device: str = "cuda",
        is_baseline: bool = False,
        train_loader=None,
        num_epochs: int = 0,
        optimizer_class=None,
        learning_rate: float = 1e-3
    ) -> AblationResult:
        """
        Run a single ablation experiment with the given configuration.
        
        This method:
        1. Creates a model using the config
        2. Optionally trains the model if train_loader and num_epochs are provided
        3. Evaluates the model on the provided loader
        4. Automatically calls add_result() with the predictions
        
        Args:
            config: AblationConfig specifying the model configuration
            loader: DataLoader for evaluation
            model_func: Function that returns a backbone model
            n_way: Number of ways (classes) in few-shot task
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            dataset: Dataset name
            device: Device to run on
            is_baseline: If True, mark this result as the baseline
            train_loader: Optional DataLoader for training
            num_epochs: Number of training epochs (0 = no training, use pretrained)
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            learning_rate: Learning rate for optimizer
        
        Returns:
            AblationResult with the experiment results
        """
        import torch
        import tqdm
        
        # Create model from config
        model = create_model_from_config(
            config=config,
            model_func=model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            dataset=dataset,
            device=device
        )
        
        # Optional training
        if train_loader is not None and num_epochs > 0:
            if optimizer_class is None:
                optimizer_class = torch.optim.Adam
            
            optimizer = optimizer_class(model.parameters(), lr=learning_rate)
            model.train()
            
            for epoch in range(num_epochs):
                for x, _ in train_loader:
                    optimizer.zero_grad()
                    _, loss = model.set_forward_loss(x.to(device))
                    loss.backward()
                    optimizer.step()
        
        # Evaluate model
        model.eval()
        all_preds = []
        all_labels = []
        acc_list = []
        
        with torch.no_grad():
            for x, _ in loader:
                scores = model.set_forward(x.to(device))
                preds = scores.data.cpu().numpy().argmax(axis=1)
                
                n_query_per_way = len(preds) // n_way
                labels = np.repeat(np.arange(n_way), n_query_per_way)
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # Calculate per-episode accuracy
                acc = np.mean(preds == labels)
                acc_list.append(acc)
        
        predictions = np.array(all_preds)
        true_labels = np.array(all_labels)
        
        # Calculate statistics
        accuracy = np.mean(acc_list)
        std = np.std(acc_list)
        
        # Calculate confidence interval
        n_episodes = len(acc_list)
        ci_margin = 1.96 * std / np.sqrt(n_episodes) if n_episodes > 0 else 0
        confidence_interval = (accuracy - ci_margin, accuracy + ci_margin)
        
        # Add result to study
        self.add_result(
            name=config.name,
            config=config,
            accuracy=accuracy,
            std=std,
            predictions=predictions,
            true_labels=true_labels,
            confidence_interval=confidence_interval,
            is_baseline=is_baseline
        )
        
        return self.results[config.name]
    
    def run_all_standard_ablations(
        self,
        loader,
        model_func,
        n_way: int,
        k_shot: int,
        n_query: int,
        dataset: str = 'miniImagenet',
        device: str = "cuda",
        train_loader=None,
        num_epochs: int = 0,
        optimizer_class=None,
        learning_rate: float = 1e-3,
        configs: Optional[List[AblationConfig]] = None,
        verbose: bool = True
    ) -> Dict[str, AblationResult]:
        """
        Run all standard ablation experiments and populate results.
        
        This convenience method runs all configurations from get_standard_ablations()
        (or custom configs if provided) and returns the complete results dictionary.
        
        Args:
            loader: DataLoader for evaluation
            model_func: Function that returns a backbone model
            n_way: Number of ways (classes) in few-shot task
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            dataset: Dataset name
            device: Device to run on
            train_loader: Optional DataLoader for training
            num_epochs: Number of training epochs (0 = no training)
            optimizer_class: Optimizer class
            learning_rate: Learning rate for optimizer
            configs: Optional list of AblationConfig objects. If None, uses get_standard_ablations()
            verbose: If True, print progress information
        
        Returns:
            Dictionary mapping config names to AblationResult objects
        """
        if configs is None:
            configs = self.get_standard_ablations()
        
        if verbose:
            print(f"Running {len(configs)} ablation configurations...")
        
        for i, config in enumerate(configs):
            if verbose:
                print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
                print(f"  Method: {config.method}, Variant: {config.variant}, "
                      f"Heads: {config.heads}, SE: {config.use_se}, "
                      f"Dynamic Weights: {config.use_dynamic_weights}")
            
            # First configuration (Full Model) is the baseline
            is_baseline = (i == 0) or (config.ablation_type == AblationType.FULL_MODEL)
            
            try:
                self.run_ablation_experiment(
                    config=config,
                    loader=loader,
                    model_func=model_func,
                    n_way=n_way,
                    k_shot=k_shot,
                    n_query=n_query,
                    dataset=dataset,
                    device=device,
                    is_baseline=is_baseline,
                    train_loader=train_loader,
                    num_epochs=num_epochs,
                    optimizer_class=optimizer_class,
                    learning_rate=learning_rate
                )
                
                if verbose:
                    result = self.results[config.name]
                    print(f"  Accuracy: {result.accuracy:.4f} ± {result.std:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                warnings.warn(f"Failed to run ablation '{config.name}': {e}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Completed {len(self.results)}/{len(configs)} ablation experiments")
            if self.baseline_result:
                print(f"Baseline ({self.baseline_name}): {self.baseline_result.accuracy:.4f}")
            print(f"{'='*60}")
        
        return self.results


# ========================================================================
# Utility Functions
# ========================================================================

def compute_performance_drop(baseline_accuracy: float, ablation_accuracy: float) -> Dict:
    """
    Compute performance drop metrics for an ablation.
    
    Args:
        baseline_accuracy: Accuracy of the baseline model
        ablation_accuracy: Accuracy of the ablated model
    
    Returns:
        Dictionary with performance metrics
    """
    absolute_drop = baseline_accuracy - ablation_accuracy
    relative_drop = (absolute_drop / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'ablation_accuracy': ablation_accuracy,
        'absolute_drop': absolute_drop,
        'relative_drop_pct': relative_drop,
        'component_contribution': absolute_drop  # Higher = more important component
    }


def rank_component_importance(ablation_results: Dict[str, Tuple[float, float]]) -> List[Dict]:
    """
    Rank components by their importance based on ablation impact.
    
    Args:
        ablation_results: Dictionary mapping ablation name to (accuracy, std)
    
    Returns:
        List of components ranked by importance
    """
    if 'baseline' not in ablation_results and 'full_model' not in ablation_results:
        raise ValueError("Results must include 'baseline' or 'full_model'")
    
    baseline_key = 'baseline' if 'baseline' in ablation_results else 'full_model'
    baseline_acc = ablation_results[baseline_key][0]
    
    rankings = []
    for name, (acc, std) in ablation_results.items():
        if name in ['baseline', 'full_model']:
            continue
        
        impact = compute_performance_drop(baseline_acc, acc)
        rankings.append({
            'component': name,
            'accuracy': acc,
            'std': std,
            **impact
        })
    
    # Sort by relative drop (descending - most important first)
    rankings.sort(key=lambda x: x['relative_drop_pct'], reverse=True)
    
    # Add rank
    for i, item in enumerate(rankings):
        item['rank'] = i + 1
    
    return rankings


def format_contingency_table(n00: int, n01: int, n10: int, n11: int,
                             algo_a_name: str = "Algorithm A",
                             algo_b_name: str = "Algorithm B") -> str:
    """
    Format contingency table as a readable string.
    
    Args:
        n00, n01, n10, n11: Contingency table values
        algo_a_name: Name of algorithm A
        algo_b_name: Name of algorithm B
    
    Returns:
        Formatted string representation
    """
    total = n00 + n01 + n10 + n11
    
    lines = [
        f"Contingency Table for McNemar's Test",
        f"",
        f"                    {algo_b_name}",
        f"                    Correct    Wrong",
        f"{algo_a_name:>12} Correct  {n11:>6}   {n10:>6}",
        f"             Wrong   {n01:>6}   {n00:>6}",
        f"",
        f"Total samples: {total}",
        f"Discordant pairs (n01 + n10): {n01 + n10}",
        f"Agreement rate: {(n00 + n11) / total * 100:.2f}%"
    ]
    
    return "\n".join(lines)


# ========================================================================
# Model Creation from Ablation Configuration
# ========================================================================

def create_model_from_config(
    config: AblationConfig,
    model_func,
    n_way: int,
    k_shot: int,
    n_query: int,
    dataset: str = 'miniImagenet',
    device: str = "cuda"
):
    """
    Create a model instance based on an AblationConfig.
    
    This function takes an AblationConfig and returns a properly configured model 
    with the specified ablation settings (variant, heads, use_se, use_dynamic_weights, etc.).
    
    Args:
        config: AblationConfig object specifying the model configuration
        model_func: Function that returns a backbone model (e.g., Conv4, ResNet18)
        n_way: Number of ways (classes) in few-shot task
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        dataset: Dataset name for backbone configuration
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        A configured model instance ready for training/evaluation
    
    Example:
        >>> config = AblationConfig(
        ...     name="Single Attention Head",
        ...     ablation_type=AblationType.SINGLE_ATTENTION_HEAD,
        ...     description="Using only 1 attention head",
        ...     method="FSCT_cosine",
        ...     variant="cosine",
        ...     heads=1,
        ...     use_dynamic_weights=True
        ... )
        >>> model = create_model_from_config(config, model_func, 5, 5, 15)
    """
    import torch
    
    method = config.method
    variant = config.variant
    heads = config.heads
    use_se = config.use_se
    use_dynamic_weights = config.use_dynamic_weights
    use_vic = config.use_vic
    
    # Handle FSCT models (FewShotTransformer)
    if method in ['FSCT_softmax', 'FSCT_cosine']:
        from methods.transformer import FewShotTransformer
        
        # Create wrapper for model_func if SE blocks need to be disabled
        if not use_se:
            backbone_func = _create_backbone_without_se(model_func, dataset)
        else:
            backbone_func = model_func
        
        model = FewShotTransformer(
            model_func=backbone_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant=variant,
            heads=heads,
            use_dynamic_weights=use_dynamic_weights
        )
    
    # Handle CTX models
    elif method in ['CTX_softmax', 'CTX_cosine']:
        from methods.CTX import CTX
        
        # Determine input dimension based on backbone
        # Default to 64 for Conv4/Conv6, 512 for ResNet
        input_dim = 64  # Default for Conv backbones
        
        model = CTX(
            model_func=model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant=variant,
            input_dim=input_dim
        )
    
    # Handle OptimalFewShot models
    elif method == 'OptimalFewShot':
        from methods.optimal_few_shot import OptimalFewShotModel
        
        model = OptimalFewShotModel(
            model_func=model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            feature_dim=64,
            n_heads=heads,
            dataset=dataset
        )
        # Apply VIC setting - OptimalFewShotModel uses VIC by default
        if not use_vic:
            # Disable VIC regularization by setting lambda to 0
            model.vic_lambda_var = 0.0
            model.vic_lambda_cov = 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: "
                        "FSCT_softmax, FSCT_cosine, CTX_softmax, CTX_cosine, OptimalFewShot")
    
    # Move model to device
    model = model.to(device)
    
    # Store config reference for later use
    model._ablation_config = config
    
    return model


def _create_backbone_without_se(original_model_func, dataset: str):
    """
    Create a wrapper model function that returns a backbone without SE blocks.
    
    For Conv4/Conv6 backbones, this returns the standard backbone since they 
    don't have SE blocks by default.
    
    For models that do have SE blocks (like OptimizedConv4 from optimal_few_shot),
    this creates a version with SE blocks disabled.
    
    Args:
        original_model_func: Original model function
        dataset: Dataset name for configuration
    
    Returns:
        A model function that returns a backbone without SE blocks
    """
    def backbone_without_se():
        model = original_model_func()
        
        # Check if the model has SE blocks that can be disabled
        if hasattr(model, 'use_se'):
            model.use_se = False
        
        # For models with encoder containing SE blocks, we can try to bypass them
        if hasattr(model, 'encoder'):
            _disable_se_in_sequential(model.encoder)
        
        return model
    
    return backbone_without_se


def _disable_se_in_sequential(sequential_module):
    """
    Recursively disable SE blocks in a sequential module by replacing their
    forward with an identity function.
    
    Args:
        sequential_module: nn.Sequential or similar module containing SE blocks
    """
    import torch.nn as nn
    
    for name, module in sequential_module.named_modules():
        # Check for SEBlock class
        if 'SEBlock' in module.__class__.__name__ or 'SqueezeExcitation' in module.__class__.__name__:
            # Replace forward with identity
            module.forward = lambda x: x


# ========================================================================
# Integration with Evaluation Utils
# ========================================================================

def extract_predictions_from_evaluation(
    loader,
    model,
    n_way: int,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract predictions and true labels from model evaluation.
    
    Args:
        loader: DataLoader for evaluation
        model: Model to evaluate
        n_way: Number of ways (classes per episode)
        device: Device to run on
    
    Returns:
        Tuple of (predictions, true_labels) as numpy arrays
    """
    import torch
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, _ in loader:
            scores = model.set_forward(x.to(device))
            preds = scores.data.cpu().numpy().argmax(axis=1)
            
            n_query = len(preds) // n_way
            labels = np.repeat(np.arange(n_way), n_query)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return np.array(all_preds), np.array(all_labels)


def compare_models_mcnemar_from_loader(
    loader,
    model_a,
    model_b,
    n_way: int,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    device: str = "cuda"
) -> Dict:
    """
    Compare two models using McNemar's test by evaluating them on a data loader.
    
    Note: This function evaluates the loader twice (once per model). For efficiency,
    consider using `extract_predictions_from_evaluation` once per model and then
    calling `mcnemar_test` directly with the cached predictions and labels.
    
    Args:
        loader: DataLoader for evaluation
        model_a: First model
        model_b: Second model
        n_way: Number of ways
        model_a_name: Name for model A
        model_b_name: Name for model B
        device: Device to run on
    
    Returns:
        McNemar's test results with model names
    """
    preds_a, labels = extract_predictions_from_evaluation(loader, model_a, n_way, device)
    preds_b, _ = extract_predictions_from_evaluation(loader, model_b, n_way, device)
    
    result = mcnemar_test(preds_a, preds_b, labels)
    result['model_a_name'] = model_a_name
    result['model_b_name'] = model_b_name
    result['model_a_accuracy'] = np.mean(preds_a == labels)
    result['model_b_accuracy'] = np.mean(preds_b == labels)
    
    return result
