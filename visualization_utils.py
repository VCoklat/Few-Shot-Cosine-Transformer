import torch
import numpy as np
import matplotlib.pyplot as plt
import types
from einops import rearrange
import torch.nn.functional as F
from methods.transformer import Attention, cosine_distance  # Import the function directly

def add_component_contribution_heatmap(module):
    """Add method to Attention class to capture component contributions"""
    # Store individual weighted components during forward pass
    module.component_contributions = None
    
    # Original forward method reference
    original_forward = module.forward
    
    def forward_with_contributions(self, q, k, v):
        # Start with original forward computation
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h = self.heads), (q, k, v))
            
        # Store output for return
        final_output = None
        
        if self.variant == "cosine":
            # Calculate all components as in original forward
            # Use cosine_distance as a standalone function, not as a method
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_scale = F.sigmoid(self.var_scale) * 3.0
            var_component = torch.matmul(q_var, k_var)
            var_component = var_component * var_scale / f_q.size(-1)
            
            if self.dynamic_weight:
                # Use global feature statistics with shot count
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                
                # IMPORTANT: Add k_shot feature like in the original code
                k_shot_feat = torch.full((self.heads, 1), float(self.k_shot) / 10.0, device=q_global.device)
                
                # Concatenate global query and key features WITH the k_shot feature
                qk_features = torch.cat([q_global, k_global, k_shot_feat], dim=-1)
                
                # print(f"qk_features shape: {qk_features.shape}, weight_predictor expects: {self.weight_predictor[0].weight.shape[1]} input features")
                
                # Predict three weights per attention head
                weights = self.weight_predictor(qk_features)
                
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))
                
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
                
                if self.record_weights and not self.training:
                    # Store weighted components for visualization
                    self.component_contributions = {
                        'cosine': (cos_weight * cosine_sim).detach().cpu().numpy(),
                        'covariance': (cov_weight * cov_component).detach().cpu().numpy(),
                        'variance': (var_weight * var_component).detach().cpu().numpy(),
                    }
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            else:
                # Fixed weights logic
                cov_weight = torch.sigmoid(self.fixed_cov_weight) 
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = 1.0 - cov_weight - var_weight
                
                if self.record_weights and not self.training:
                    # Store weighted components for visualization
                    self.component_contributions = {
                        'cosine': (cos_weight * cosine_sim).detach().cpu().numpy(),
                        'covariance': (cov_weight * cov_component).detach().cpu().numpy(),
                        'variance': (var_weight * var_component).detach().cpu().numpy(),
                    }
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            
            # Store attention scores for rollout visualization
            if hasattr(self, 'record_attention') and self.record_attention:
                # Store full attention matrix, not just a single position
                self.attention_scores = dots.detach().clone()
                
            out = torch.matmul(dots, f_v)
            final_output = out
        else:
            # Original softmax implementation
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale  
            
            # Store attention scores for rollout visualization
            if hasattr(self, 'record_attention') and self.record_attention:
                # Store full attention matrix, not just a single position
                self.attention_scores = dots.detach().clone()
                
            out = torch.matmul(self.sm(dots), f_v)
            final_output = out
            
        out = rearrange(final_output, 'h q n d -> q n (h d)')
        return self.output_linear(out)
            
    # Replace the forward method
    module.forward = types.MethodType(forward_with_contributions, module)
    
    # Add visualization method
    def visualize_component_contributions(self, figsize=(15, 5)):
        """Visualize the contribution of each component as a heatmap"""
        if not hasattr(self, 'component_contributions') or self.component_contributions is None:
            print("No component contributions recorded. Set record_weights=True and run inference")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        components = ['cosine', 'covariance', 'variance']
        
        for i, comp in enumerate(components):
            # Average across heads for visualization
            avg_contrib = np.mean(self.component_contributions[comp], axis=0)
            im = axes[i].imshow(avg_contrib[0], aspect='auto', cmap='viridis')
            axes[i].set_title(f"{comp.capitalize()} Contribution")
            axes[i].set_xlabel("Key Position")
            axes[i].set_ylabel("Query Position")
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        return fig
    
    module.visualize_component_contributions = types.MethodType(visualize_component_contributions, module)


def add_weight_radar_chart(module):
    """Add method to Attention class to visualize weight distribution as radar chart"""
    def visualize_weight_radar(self, figsize=(10, 8)):
        """Create a radar chart showing weight distribution across heads"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle, RegularPolygon
        from matplotlib.path import Path
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        if not self.weight_history:
            print("No weights recorded. Set record_weights=True and run inference")
            return None
        
        # Define radar chart function (implementation omitted for brevity)
        # ... radar chart implementation code ...
        
        # Process weights
        weights = np.array(self.weight_history)
        if weights.shape[1] != 3:
            print("Weight history doesn't have 3 components. Format not supported.")
            return None
        
        # Create radar chart
        labels = ['Cosine', 'Covariance', 'Variance']
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Calculate average weights
        avg_weights = weights.mean(axis=0)
        
        # Plot the weights on the radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        avg_weights = np.concatenate((avg_weights, [avg_weights[0]]))  # Close the polygon
        
        ax.plot(angles, avg_weights, 'o-', linewidth=2)
        ax.fill(angles, avg_weights, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax.set_ylim(0, 1)
        
        plt.title('Component Weight Distribution')
        
        return fig
    
    module.visualize_weight_radar = types.MethodType(visualize_weight_radar, module)


def add_weight_evolution_tracking(model):
    """Set up weight evolution tracking for a model with Attention modules"""
    # Initialize storage for weights across epochs
    model.epoch_weight_history = []
    model.epoch_var_scale_history = []
    
    # Method to record weights at end of an epoch
    def record_epoch_weights(self, epoch):
        # Collect weights from all attention modules
        epoch_weights = []
        epoch_var_scales = []
        
        for module in self.modules():
            if isinstance(module, Attention) and hasattr(module, 'dynamic_weight'):
                if module.weight_history:
                    # Average the weights collected during this epoch
                    avg_weights = np.mean(np.array(module.weight_history), axis=0)
                    epoch_weights.append((epoch, avg_weights))
                    
                # Record the current var_scale value
                if hasattr(module, 'var_scale'):
                    var_scale_value = float(F.sigmoid(module.var_scale).item() * 3.0)
                    epoch_var_scales.append((epoch, var_scale_value))
                
                # Clear the within-epoch history for the next epoch
                module.clear_weight_history()
        
        if epoch_weights:
            self.epoch_weight_history.extend(epoch_weights)
        
        if epoch_var_scales:
            self.epoch_var_scale_history.extend(epoch_var_scales)
    
    # Add method to model
    model.record_epoch_weights = types.MethodType(record_epoch_weights, model)
    
    # Method to visualize weight evolution
    def plot_weight_evolution(self, figsize=(10, 6)):
        if not hasattr(self, 'epoch_weight_history') or not self.epoch_weight_history:
            print("No weight evolution data recorded.")
            return None
        
        # Extract data
        epochs = sorted(list(set([entry[0] for entry in self.epoch_weight_history])))
        
        # Aggregate weights by epoch (may have multiple attention modules)
        epoch_weights = {}
        for epoch in epochs:
            epoch_weights[epoch] = []
            
        for epoch_num, weights in self.epoch_weight_history:
            epoch_weights[epoch_num].append(weights)
        
        # Average across modules for each epoch
        avg_weights = {epoch: np.mean(weights, axis=0) for epoch, weights in epoch_weights.items()}
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each component
        components = ['Cosine', 'Covariance', 'Variance']
        markers = ['o', 's', '^']
        
        for i, comp in enumerate(components):
            values = [avg_weights[epoch][i] for epoch in epochs]
            ax.plot(epochs, values, f'-{markers[i]}', label=comp)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title('Attention Component Weight Evolution')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    # Add visualization method to model
    model.plot_weight_evolution = types.MethodType(plot_weight_evolution, model)
    
    # Method to visualize variance scale evolution
    def plot_var_scale_evolution(self, figsize=(10, 6)):
        if not hasattr(self, 'epoch_var_scale_history') or not self.epoch_var_scale_history:
            print("No var_scale evolution data recorded.")
            return None
        
        # Extract data
        epochs = []
        var_scales = []
        
        # May have multiple modules reporting var_scale
        epoch_var_scales = {}
        for epoch_num, var_scale in self.epoch_var_scale_history:
            if epoch_num not in epoch_var_scales:
                epoch_var_scales[epoch_num] = []
            epoch_var_scales[epoch_num].append(var_scale)
        
        # Average across modules for each epoch
        epochs = sorted(epoch_var_scales.keys())
        var_scales = [np.mean(epoch_var_scales[epoch]) for epoch in epochs]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, var_scales, 'o-', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance Scaling Factor')
        ax.set_title('Variance Scaling Factor Evolution')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    # Add visualization method to model
    model.plot_var_scale_evolution = types.MethodType(plot_var_scale_evolution, model)


def visualize_attention_rollout(model, val_loader, save_path=None):
    """Visualize attention scores for few-shot episodes
    
    Args:
        model: FewShotTransformer model
        val_loader: Validation data loader containing few-shot episodes
        save_path: Path to save the visualization (None for no saving)
    
    Returns:
        matplotlib figure with attention visualizations
    """
    device = next(model.parameters()).device
    
    # Enable attention recording
    for module in model.modules():
        if isinstance(module, Attention):
            module.record_attention = True
            if not hasattr(module, 'attention_scores'):
                module.attention_scores = None
    
    # Process a proper few-shot episode through the model
    model.eval()
    with torch.no_grad():
        # Get a single episode from the loader
        x, _ = next(iter(val_loader))
        x = x.to(device)
        # Forward pass
        _ = model.set_forward(x)
    
    # Create visualization
    figs = []
    
    for i, module in enumerate(model.modules()):
        if isinstance(module, Attention) and hasattr(module, 'attention_scores') and module.attention_scores is not None:
            attention = module.attention_scores.cpu().numpy()
            
            # Debug the attention tensor shape
            print(f"Attention shape for module {i}: {attention.shape}")
            
            # Skip if the shape isn't suitable for visualization
            if len(attention.shape) < 3:
                print(f"Skipping module {i} - attention shape not suitable for visualization")
                continue
                
            # Create figure for this attention module
            n_heads = attention.shape[0]
            fig, axes = plt.subplots(1, n_heads, figsize=(n_heads*3, 3))
            if n_heads == 1:
                axes = [axes]
            
            for h in range(n_heads):
                # Select the full attention matrix for this head
                # Instead of trying to get a specific position which might be a scalar
                if len(attention.shape) == 4:  # [heads, query_batch, seq_len1, seq_len2]
                    # Get the first query's attention pattern
                    att_matrix = attention[h, 0]
                elif len(attention.shape) == 3:  # [heads, seq_len1, seq_len2]
                    att_matrix = attention[h]
                else:
                    print(f"Unexpected attention shape: {attention.shape}")
                    continue
                    
                # Ensure we have a 2D matrix
                if att_matrix.ndim < 2:
                    print(f"Head {h} attention is not 2D, shape: {att_matrix.shape}")
                    continue
                
                im = axes[h].imshow(att_matrix, cmap='viridis')
                axes[h].set_title(f'Head {h+1}')
                axes[h].set_xlabel('Key Position')
                axes[h].set_ylabel('Query Position')
                fig.colorbar(im, ax=axes[h])
            
            plt.tight_layout()
            plt.suptitle(f'Attention Module {i}', y=1.05)
            
            if save_path:
                plt.savefig(f"{save_path}_module_{i}.png")
            
            figs.append(fig)
            
            # Clear recorded attention
            module.attention_scores = None
    
    # Disable attention recording
    for module in model.modules():
        if isinstance(module, Attention) and hasattr(module, 'record_attention'):
            module.record_attention = False
    
    return figs


def analyze_class_specific_weights(model, data_loader, n_classes, save_path=None):
    """Collect and visualize class-specific attention weight distributions"""
    device = next(model.parameters()).device
    
    # Storage for class-specific weights
    class_weights = {c: {'cosine': [], 'covariance': [], 'variance': []} for c in range(n_classes)}
    
    # Set model to eval mode and enable weight recording
    model.eval()
    for module in model.modules():
        if isinstance(module, Attention):
            module.record_weights = True
            module.clear_weight_history()
    
    # Process batches
    with torch.no_grad():
        batch_count = 0
        for x, y in data_loader:
            if batch_count >= 5:  # Limit to 5 batches to speed up analysis
                break
                
            x = x.to(device)
            
            # Forward pass to collect weights
            _ = model.set_forward(x)
            
            # Get weights from attention module
            for module in model.modules():
                if isinstance(module, Attention) and module.weight_history:
                    # In few-shot learning, we might not have explicit class labels in y
                    # Instead, we'll use position in the batch as pseudo-class
                    n_samples = min(len(module.weight_history), n_classes)
                    
                    for i in range(n_samples):
                        # Assign class based on position (simplified approach)
                        class_idx = i % n_classes
                        
                        if i < len(module.weight_history):
                            weights = module.weight_history[i]
                            class_weights[class_idx]['cosine'].append(weights[0])
                            class_weights[class_idx]['covariance'].append(weights[1])
                            class_weights[class_idx]['variance'].append(weights[2])
            
            # Clear history after each batch
            for module in model.modules():
                if isinstance(module, Attention):
                    module.clear_weight_history()
                    
            batch_count += 1
    
    # Turn off weight recording
    for module in model.modules():
        if isinstance(module, Attention):
            module.record_weights = False
    
    # Create box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['cosine', 'covariance', 'variance']
    
    # Filter out empty classes
    active_classes = [c for c in range(n_classes) if class_weights[c]['cosine']]
    
    if not active_classes:
        print("No class data collected. Check if labels are within the expected range.")
        # Create empty plot with message
        for i, comp in enumerate(components):
            axes[i].text(0.5, 0.5, "No class data available", 
                         horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(f'{comp.capitalize()} Weight Distribution')
    else:
        print(f"Found data for {len(active_classes)} classes: {active_classes}")
        for i, comp in enumerate(components):
            data = [class_weights[c][comp] for c in active_classes]
            axes[i].boxplot(data)
            axes[i].set_title(f'{comp.capitalize()} Weight Distribution')
            axes[i].set_xlabel('Position in Episode')
            axes[i].set_ylabel('Weight Value')
            axes[i].set_xticklabels([str(c) for c in active_classes])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return class_weights, fig


def setup_visualization_tools(model):
    """Set up all visualization tools for a model"""
    # Add component contribution heatmap
    for module in model.modules():
        if isinstance(module, Attention):
            add_component_contribution_heatmap(module)
            add_weight_radar_chart(module)
            # Add record_attention attribute
            module.record_attention = False
    
    # Add weight evolution tracking
    add_weight_evolution_tracking(model)
    
    return model


def enable_weight_recording(model, enable=True):
    """Enable or disable weight recording for all attention modules"""
    for module in model.modules():
        if isinstance(module, Attention):
            module.record_weights = enable
            if enable:
                module.clear_weight_history()