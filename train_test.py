import glob
import json
import os
import pdb
import pprint
import random
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.sampler
import tqdm
from torch.autograd import Variable
from torchsummary import summary

# Additional imports for enhanced functionality
try:
    import psutil
except ImportError:
    print("psutil not installed. System monitoring will be limited.")
    psutil = None

try:
    import GPUtil
except ImportError:
    print("GPUtil not installed. GPU monitoring will be limited.")
    GPUtil = None

from sklearn.metrics import confusion_matrix

import backbone
import configs
import data.feature_loader as feat_loader
import wandb
from data.datamgr import SetDataManager
from io_utils import (get_assigned_file, get_best_file,
                     model_dict, parse_args)
from methods.CTX import CTX
from methods.transformer import FewShotTransformer
from methods.optimal_few_shot import OptimalFewShotModel, DATASET_CONFIGS
import eval_utils

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretty_print(res):
    """Enhanced pretty printing"""
    if not res:
        print("No results to display!")
        return

    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 50)
    # F1 metric removed — show core metrics instead

    print("\n🔢 Confusion matrix:")
    print(np.array(res["conf_mat"]))

    print(f"\n⏱️ Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"💾 Model size: {res['param_count']:.2f} M params")
    print(f"🖥️ GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"🖥️ CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    print("=" * 50)

def get_class_names_from_file(data_file, n_way=None):
    """
    Generate class names for few-shot evaluation.
    
    In few-shot learning, each episode randomly samples n_way classes from the dataset,
    and labels are re-mapped to 0, 1, ..., n_way-1 for each episode. Therefore, the
    class names should be generic labels representing these positions, not specific
    class names from the dataset (which vary across episodes).
    """
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()
        total_classes = len(unique_labels)
        
        # In few-shot learning, labels are re-mapped to 0 to n_way-1 for each episode
        # Use generic names that represent the position/way rather than specific classes
        if n_way:
            class_names = [f"Way {i}" for i in range(n_way)]
        else:
            # If n_way is not specified, use all available classes
            class_names = [f"Class {i}" for i in range(total_classes)]
        
        print(f"Dataset has {total_classes} classes total, using {len(class_names)} ways for evaluation")
        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Way {i}" for i in range(n_way)] if n_way else ["Class_0"]

def get_system_stats():
    """Get system resource utilization stats"""
    # Default values
    cpu_percent = 0.0
    cpu_mem_used_MB = 0.0
    cpu_mem_total_MB = 0.0
    gpu_util = 0.0
    gpu_mem_used_MB = 0
    gpu_mem_total_MB = 0
    
    # CPU stats
    if psutil:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            cpu_mem_used_MB = memory.used / (1024**2)
            cpu_mem_total_MB = memory.total / (1024**2)
        except:
            pass
    
    # GPU stats
    try:
        if GPUtil and torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_util = gpu.load
                gpu_mem_used_MB = gpu.memoryUsed
                gpu_mem_total_MB = gpu.memoryTotal
    except:
        # Fallback to torch methods
        try:
            if torch.cuda.is_available():
                gpu_mem_used_MB = torch.cuda.memory_allocated() / (1024**2)
                gpu_mem_total_MB = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        except:
            pass
    
    return {
        'cpu_util': cpu_percent,
        'cpu_mem_used_MB': cpu_mem_used_MB,
        'cpu_mem_total_MB': cpu_mem_total_MB,
        'gpu_util': gpu_util,
        'gpu_mem_used_MB': gpu_mem_used_MB,
        'gpu_mem_total_MB': gpu_mem_total_MB
    }

def evaluate_comprehensive(test_loader, model, params, data_file):
    """Comprehensive evaluation with enhanced metrics and system monitoring"""
    model.eval()
    
    # Get class names
    class_names = get_class_names_from_file(data_file, params.n_way)
    
    # Initialize metrics
    all_predictions = []
    all_true_labels = []
    inference_times = []
    
    # Model parameter count
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"Running comprehensive evaluation on {len(test_loader)} episodes...")
    
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader)) as pbar:
            for i, (x, _) in enumerate(test_loader):
                # Measure inference time
                start_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=True):
                    scores = model.set_forward(x)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions
                pred = scores.data.cpu().numpy().argmax(axis=1)
                y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
                
                all_predictions.extend(pred)
                all_true_labels.extend(y)
                
                # Clear memory
                del scores
                torch.cuda.empty_cache()
                
                pbar.set_description(f'Evaluating episode {i+1}/{len(test_loader)}')
                pbar.update(1)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate comprehensive metrics
    conf_mat = confusion_matrix(all_true_labels, all_predictions)
    # F1 score computation removed for streamlined metrics
    
    # System stats
    system_stats = get_system_stats()
    
    # Compile results
    results = {
        'class_names': class_names,
        'conf_mat': conf_mat.tolist(),
        'avg_inf_time': np.mean(inference_times),
        'param_count': param_count,
        **system_stats
    }
    
    return results

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(num_epoch):
        model.train()
        model.train_loop(epoch, num_epoch, base_loader,
                        params.wandb, optimizer)

        with torch.no_grad():
            model.eval()

            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
        print()
    return model

def direct_test(test_loader, model, params, data_file=None, comprehensive=True, n_way=None):
    """Enhanced testing function with optional comprehensive evaluation
    
    Args:
        test_loader: DataLoader for test episodes
        model: Model to evaluate
        params: Parameters object
        data_file: Path to data file (for class names)
        comprehensive: If True, use comprehensive evaluation
        n_way: Number of ways for evaluation. If None, uses params.n_way
    """
    # Use provided n_way or fall back to params.n_way
    eval_n_way = n_way if n_way is not None else params.n_way
    
    if comprehensive and data_file:
        # Get class names from data file
        class_names = get_class_names_from_file(data_file, eval_n_way)
        
        # Use eval_utils comprehensive evaluation. If feature_analysis is requested,
        # call the comprehensive variant which runs the feature space metrics.
        if getattr(params, 'feature_analysis', False):
            results = eval_utils.evaluate_comprehensive(test_loader, model, eval_n_way,
                                                       class_names=class_names, device=device)
            eval_utils.pretty_print(results, show_feature_analysis=True)
            # Explicitly print key feature-analysis fields for quick visibility
            if 'feature_analysis' in results and results['feature_analysis']:
                fa = results['feature_analysis']
                try:
                    print("\n🔎 Feature Analysis Summary:")
                    print(f"  Collapsed dimensions: {fa['feature_collapse']['collapsed_dimensions']}/{fa['feature_collapse']['total_dimensions']}")
                    print(f"  Mean utilization: {fa['feature_utilization']['mean_utilization']:.4f} (low dims: {fa['feature_utilization']['low_utilization_dims']})")
                    print(f"  Mean diversity (CV): {fa['diversity_score']['mean_diversity']:.4f}")
                    print(f"  Effective dimensions (95%): {fa['feature_redundancy']['effective_dimensions_95pct']} of {fa['feature_redundancy']['total_features']}")
                    print(f"  High correlation pairs (>0.9): {fa['feature_redundancy']['high_correlation_pairs']}")
                    print(f"  Mean intra-class euclidean consistency: {fa['intraclass_consistency']['mean_euclidean_consistency']:.4f}")
                    print(f"  Mean intra-class cosine consistency: {fa['intraclass_consistency']['mean_cosine_consistency']:.4f}")
                    print(f"  Imbalance ratio (minor/major): {fa['imbalance_ratio']['imbalance_ratio']:.3f}")
                    print("  Most confusing pairs (top):")
                    for pair in fa['confusing_pairs']['most_confusing_pairs']:
                        print(f"    {pair['class_1']} ↔ {pair['class_2']}: distance = {pair['distance']:.4f}")
                except Exception as e:
                    print(f"Error printing feature analysis summary: {e}")
        else:
            results = eval_utils.evaluate(test_loader, model, eval_n_way,
                                          class_names=class_names, device=device)
            eval_utils.pretty_print(results, show_feature_analysis=False)
        
        # Still return traditional accuracy metrics for backward compatibility
        acc_mean = results['accuracy'] * 100  # Convert accuracy to percentage scale
        # Use episode-level standard deviation for variability (was previously class-F1 std)
        acc_std = np.std(results.get('episode_accuracies', [acc_mean/100]))*100
        
        return acc_mean, acc_std, results
    else:
        # Original direct_test functionality with F1 score calculation
        acc = []
        all_preds = []
        all_labels = []
        iter_num = len(test_loader)
        
        if iter_num == 0:
            print("ERROR: Test loader is empty")
            return 0.0, 0.0

        with tqdm.tqdm(total=len(test_loader)) as pbar:
            for i, (x, _) in enumerate(test_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    scores = model.set_forward(x)
                    pred = scores.data.cpu().numpy().argmax(axis=1)
                    
                    # Move computation to CPU and free GPU memory
                    y = np.repeat(range(eval_n_way), pred.shape[0]//eval_n_way)
                    
                    all_preds.extend(pred.tolist())
                    all_labels.extend(y.tolist())
                    
                    acc.append(np.mean(pred == y)*100)
                    
                    # Clear unnecessary tensors
                    del scores
                    torch.cuda.empty_cache()
                    
                    pbar.set_description('Test | Acc {:.6f}'.format(np.mean(acc)))
                    pbar.update(1)
        
        acc_all = np.asarray(acc)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        
        # F1 score calculation removed — keep accuracy only
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # report accuracy only
        
        return acc_mean, acc_std

def seed_func():
    seed = 4040
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(10)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def change_model(model_name):
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

if __name__ == '__main__':
    params = parse_args()
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()

    project_name = "Few-Shot_TransFormer"

    if params.dataset == 'Omniglot': 
        params.n_query = 15

    if params.wandb:
        wandb_name = params.method + "_" + params.backbone + "_" + params.dataset + \
                    "_" + str(params.n_way) + "w" + str(params.k_shot) + "s"
        if params.train_aug:
            wandb_name += "_aug"
        if params.FETI and 'ResNet' in params.backbone:
            wandb_name += "_FETI"
        wandb_name += "_" + params.datetime
        wandb.init(project=project_name, name=wandb_name,
                  config=params, id=params.datetime)

    print()

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84

    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4': params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': params.backbone = 'Conv6S'

    optimization = params.optimization

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine', 'OptimalFewShot']:
        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query = params.n_query)
        
        base_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)

        val_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(
            val_file, aug=False)

        seed_func()

        if params.method in ['FSCT_softmax', 'FSCT_cosine']:
            variant = 'cosine' if params.method == 'FSCT_cosine' else 'softmax'
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)
            
            model = FewShotTransformer(
                feature_model, 
                variant=variant,
                heads=params.num_heads,
                use_dynamic_weights=bool(params.use_dynamic_weights),
                **few_shot_params
            )

        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)
            
            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)

        elif params.method == 'OptimalFewShot':
            # Get dataset-specific configuration
            config = DATASET_CONFIGS.get(params.dataset, DATASET_CONFIGS['miniImagenet'])
            
            # Create feature model function based on backbone parameter
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)
            
            # Create model with dataset-specific parameters
            use_focal_loss = config.get('focal_loss', False)
            dropout = config.get('dropout', 0.1)
            
            model = OptimalFewShotModel(
                feature_model,
                n_way=params.n_way,
                k_shot=params.k_shot,
                n_query=params.n_query,
                feature_dim=64,
                n_heads=4,
                dropout=dropout,
                num_datasets=5,
                dataset=params.dataset,
                use_focal_loss=use_focal_loss,
                label_smoothing=0.1
            )

        else:
            raise ValueError('Unknown method')

        model = model.to(device)

        params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.backbone, params.method)

        if params.train_aug:
            params.checkpoint_dir += '_aug'
        if params.FETI and 'ResNet' in params.backbone:
            params.checkpoint_dir += '_FETI'

        params.checkpoint_dir += '_%dway_%dshot' % (
            params.n_way, params.k_shot)

        # Add ablation parameters to checkpoint path for FSCT models
        if params.method in ['FSCT_softmax', 'FSCT_cosine']:
            params.checkpoint_dir += '_heads%d' % params.num_heads
            if not params.use_dynamic_weights:
                params.checkpoint_dir += '_nodyn'

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        print("===================================")
        print("Train phase: ")
        model = train(base_loader, val_loader, model, optimization, params.num_epoch, params)

        ######################################################################
        print("===================================")
        print("Test phase: ")
        
        # Memory optimization
        torch.cuda.empty_cache()
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        iter_num = params.test_iter
        split = params.split

        if params.dataset == 'cross':
            if split == 'base':
                testfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                testfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
            else:
                testfile = configs.data_dir['emnist'] + split + '.json'
        else:
            testfile = configs.data_dir[params.dataset] + split + '.json'

        print(f"Testing with file: {testfile}")

        if not os.path.exists(testfile):
            print(f"ERROR: Test file {testfile} does not exist!")

        if params.save_iter != -1:
            modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(params.checkpoint_dir)

        # Check the number of classes in the test dataset and adjust n_way if necessary
        test_n_way = params.n_way
        try:
            with open(testfile, 'r') as f:
                test_meta = json.load(f)
            test_classes = len(np.unique(test_meta['image_labels']))
            if test_classes < params.n_way:
                print(f"WARNING: Test dataset has only {test_classes} classes but n_way={params.n_way}.")
                print(f"Adjusting n_way to {test_classes} for testing.")
                test_n_way = test_classes
        except Exception as e:
            print(f"Warning: Could not check test dataset classes: {e}")

        # Use adjusted n_way for test data manager
        test_few_shot_params = dict(
            n_way=test_n_way, k_shot=params.k_shot, n_query=params.n_query)
        
        test_datamgr = SetDataManager(
            image_size, n_episode=iter_num, **test_few_shot_params)
        test_loader = test_datamgr.get_data_loader(testfile, aug=False)

        # Add this before calling direct_test
        test_loader_size = len(test_loader)
        print(f"Test loader contains {test_loader_size} episodes")

        if test_loader_size == 0:
            print("WARNING: Test loader is empty! Check your data paths and configuration.")

        model = model.to(device)

        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])

        # Update model's n_way if test dataset has fewer classes
        if test_n_way != params.n_way:
            if hasattr(model, 'n_way'):
                print(f"Updating model n_way from {model.n_way} to {test_n_way}")
                model.n_way = test_n_way

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        # Enhanced test execution with comprehensive evaluation option
        if params.comprehensive_eval:
            # Use comprehensive evaluation
            acc_mean, acc_std, detailed_results = direct_test(
                test_loader, model, params, testfile, comprehensive=True, n_way=test_n_way
            )
            
            # Log detailed results to wandb if available
            if params.wandb:
                wandb.log({
                    'Test Acc': acc_mean,
                    'Avg Inference Time': detailed_results['avg_inf_time'],
                    'Model Size (M params)': detailed_results['param_count']
                })
        else:
            # Use standard evaluation
            acc_mean, acc_std = direct_test(test_loader, model, params, n_way=test_n_way)

        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

        if params.wandb:
            wandb.log({'Test Acc': acc_mean})

        with open('./record/results.txt', 'a') as f:
            timestamp = params.datetime
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-FETI' if params.FETI and 'ResNet' in params.backbone else ''

            if params.backbone == "Conv4SNP":
                params.backbone = "Conv4"
            elif params.backbone == "Conv6SNP":
                params.backbone = "Conv6"

            exp_setting = '%s-%s-%s%s-%sw%ss' % (params.dataset, params.backbone,
                                               params.method, aug_str, params.n_way, params.k_shot)
            acc_str = 'Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std/np.sqrt(iter_num))
            f.write('Time: %s Setting: %s %s \n' % (timestamp, exp_setting.ljust(50), acc_str))

        if params.wandb:
            wandb.finish()
