import glob
import json
import os
import pdb
import pprint
import random
import timef
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

from sklearn.metrics import confusion_matrix, f1_score

import backbone
import configs
import data.feature_loader as feat_loader
import wandb
from data.datamgr import SetDataManager
from io_utils import (get_assigned_file, get_best_file,
                     model_dict, parse_args)
from methods.CTX import CTX
from methods.transformer import FewShotTransformer

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretty_print(res):
    """Enhanced pretty printing"""
    if not res:
        print("No results to display!")
        return

    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 50)
    print(f"🎯 Macro-F1: {res['macro_f1']:.4f}")

    print("\n📈 Per-class F1 scores:")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

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
    """Extract class names from JSON data file"""
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()

        if 'class_names' in meta:
            class_names = [meta['class_names'][str(label)] for label in unique_labels]
        else:
            class_names = [f"Class_{label}" for label in unique_labels]

        if n_way and len(class_names) > n_way:
            class_names = class_names[:n_way]

        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Class_{i}" for i in range(n_way)] if n_way else ["Class_0"]

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
    class_f1 = f1_score(all_true_labels, all_predictions, average=None)
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    
    # System stats
    system_stats = get_system_stats()
    
    # Compile results
    results = {
        'macro_f1': macro_f1,
        'class_f1': class_f1.tolist(),
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

def direct_test(test_loader, model, params, data_file=None, comprehensive=True):
    """Enhanced testing function with optional comprehensive evaluation"""
    if comprehensive and data_file:
        # Use comprehensive evaluation
        results = evaluate_comprehensive(test_loader, model, params, data_file)
        pretty_print(results)
        
        # Still return traditional accuracy metrics for backward compatibility
        acc_mean = results['macro_f1'] * 100  # Convert F1 to percentage scale
        acc_std = np.std([f1 * 100 for f1 in results['class_f1']])  # Std of class F1s
        
        return acc_mean, acc_std, results
    else:
        # Original direct_test functionality
        acc = []
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
                    y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
                    acc.append(np.mean(pred == y)*100)
                    
                    # Clear unnecessary tensors
                    del scores
                    torch.cuda.empty_cache()
                    
                    pbar.set_description('Test | Acc {:.6f}'.format(np.mean(acc)))
                    pbar.update(1)
        
        acc_all = np.asarray(acc)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        
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
    
    # Add comprehensive evaluation parameter if not present
    if not hasattr(params, 'comprehensive_eval'):
        params.comprehensive_eval = True
    
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

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine']:
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
            
            model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)

        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)
            
            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)

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

        test_datamgr = SetDataManager(
            image_size, n_episode=iter_num, **few_shot_params)
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

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        # Enhanced test execution with comprehensive evaluation option
        if params.comprehensive_eval:
            # Use comprehensive evaluation
            acc_mean, acc_std, detailed_results = direct_test(
                test_loader, model, params, testfile, comprehensive=True
            )
            
            # Log detailed results to wandb if available
            if params.wandb:
                wandb.log({
                    'Test Acc': acc_mean,
                    'Macro F1': detailed_results['macro_f1'],
                    'Avg Inference Time': detailed_results['avg_inf_time'],
                    'Model Size (M params)': detailed_results['param_count']
                })
        else:
            # Use standard evaluation
            acc_mean, acc_std = direct_test(test_loader, model, params)

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
