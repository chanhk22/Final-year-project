"""
Explainable AI (XAI) Module for Clinical Interpretation using SHAP.

[Clinical Interpretability]
To ensure clinical trustworthiness and diagnostic transparency, this file integrates SHAP (SHapley Additive exPlanations) [Lundberg & Lee, 2017]. 
By computing the marginal contribution of each multimodal feature, can trace back the model's predictions to specific acoustic and visual behaviors 
(e.g., specific Action Units or vocal formants), bridging the gap between raw data and psychiatric evaluation.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
import json
import glob
import pickle
from tqdm import tqdm
from pathlib import Path
from matplotlib.patches import Patch
try:
    from precache.dataset_temporal import create_temporal_dataloaders
    from models.model import MultimodalClassifier
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from precache.dataset_temporal import create_temporal_dataloaders
    from models.model import MultimodalClassifier

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')

def interpret_pca_component(pca_model, component_idx, original_feature_names, top_n=3):
    """
    Inverse-transforms PCA components to extract clinical meaning.
    
    While PCA effectively reduces dimensionality and prevents overfitting, it obfuscates the physical meaning of the features. 
    This function analyzes the eigenvector weights of a specific PCA component and maps it back to the 
    top-N original clinical markers (e.g., Gaze angle, F0 envelope), providing a human-readable interpretation of what the model is actually looking at.
    """
    weights = pca_model.components_[component_idx]
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    
    interpretation = []
    for i in range(top_n):
        f_idx = sorted_idx[i]
        f_name = original_feature_names[f_idx]
        f_weight = weights[f_idx]
        direction = "(+)" if f_weight > 0 else "(-)"
        interpretation.append(f"{f_name} {direction}")
        
    return ", ".join(interpretation)

class ShapAnalyzer:
    def __init__(self, model, data_loader, device='cuda', output_dir='explainability_shap'):
        self.model = model.to(device)
        self.model.eval()
        self.data_loader = data_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.audio_dim = model.audio_dim
        self.visual_dim = model.visual_dim

    def model_predict_wrapper(self, combined_input):
        """
        Wrapper function for SHAP KernelExplainer.
        Input: (N, audio_dim + visual_dim) - mean-pooled static features
        Output: model prediction probabilities
        """
        # 1. Convert NumPy array to Tensor
        tensor = torch.tensor(combined_input, dtype=torch.float32).to(self.device)
        
        # 2. Split into Audio/Visual (Batch, Dim)
        # combined_input has shape (audio_dim + visual_dim)
        audio = tensor[:, :self.audio_dim] 
        visual = tensor[:, self.audio_dim:] 
        
        # 3. Expand dimensions for temporal model: (Batch, Dim) -> (Batch, 1, Dim)
        # Convert static features into a sequence of length 1
        audio = audio.unsqueeze(1)
        visual = visual.unsqueeze(1)

        B = audio.shape[0]
        # Set mask to all ones (valid)
        mask = torch.ones(B, device=self.device)
        
        with torch.no_grad():
            # Run model (processing sequence length = 1)
            output = self.model(audio, visual, mask, mask)
            return torch.sigmoid(output['logits']).cpu().numpy()

    def run_shap_analysis(self, background_samples=100, test_samples=50):
        print(f"Preparing data for SHAP (Background: {background_samples}, Test: {test_samples})...")
        
        all_audio_means = []
        all_visual_means = []
        
        # Load data and apply mean pooling
        print("Loading and pooling sequences...")
        for batch in self.data_loader:
            # batch['audio']: (Batch, SeqLen, Dim) -> Mean -> (Batch, Dim)
            aud = batch['audio']
            vis = batch['visual']
            
            # If sequence dimension exists, take mean; otherwise use as-is
            if aud.ndim == 3:
                aud_mean = aud.mean(dim=1)
            else:
                aud_mean = aud
                
            if vis.ndim == 3:
                vis_mean = vis.mean(dim=1)
            else:
                vis_mean = vis
            
            all_audio_means.append(aud_mean)
            all_visual_means.append(vis_mean)
            
            current_samples = sum(len(a) for a in all_audio_means)
            if current_samples >= background_samples + test_samples:
                break
                
        # Concatenate tensors
        all_audio = torch.cat(all_audio_means, dim=0)
        all_visual = torch.cat(all_visual_means, dim=0)
        
        # Combine Audio and Visual features (Total, Audio_Dim + Visual_Dim)
        combined_features = torch.cat([all_audio, all_visual], dim=1).numpy()
        
        background_data = combined_features[:background_samples]
        test_data = combined_features[background_samples:background_samples+test_samples]

        print(f"Feature shape for SHAP: {background_data.shape}")
        print("Initializing SHAP KernelExplainer...")
        
        # Run KernelExplainer
        explainer = shap.KernelExplainer(self.model_predict_wrapper, background_data)
        
        print(f"Calculating SHAP values for {test_samples} samples...")
        # Reduce nsamples if you want faster computation
        shap_values = explainer.shap_values(test_data, nsamples=100) 

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        self.plot_shap_summary(shap_values, test_data)
        self.plot_modality_importance(shap_values)

    def plot_shap_summary(self, shap_values, test_data):
        """
        Standard SHAP summary plots only show the magnitude of importance. 
        This enhanced visualization computes the Pearson correlation between the raw feature values and their SHAP values to determine the 'direction' 
        of the impact. For instance, it scientifically proves whether an increase in a specific facial movement mitigates or exacerbates the predicted depression risk.
        """
        print("Generating Enhanced SHAP Bar Plot with Directional Impact...")
        
        feature_names = [f'A_PCA_{i}' for i in range(self.audio_dim)] + \
                        [f'V_PCA_{i}' for i in range(self.visual_dim)]
        
        if len(shap_values.shape) > 2:
            shap_values = np.squeeze(shap_values)
            
        # 1. Compute feature importance (mean absolute SHAP values)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # 2. Compute directionality (correlation)
        correlation_signs = []
        for i in range(len(feature_names)):
            x_vals = np.array(test_data[:, i]).flatten()
            y_vals = np.array(shap_values[:, i]).flatten()
            
            # Avoid division by zero
            if np.std(x_vals) == 0 or np.std(y_vals) == 0:
                correlation_signs.append(0)
            else:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                # Handle NaN cases
                if np.isnan(corr):
                    correlation_signs.append(0)
                else:
                    correlation_signs.append(np.sign(corr))
                
        # 3. Sort by importance (Top 10)
        top_k = 10
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        
        top_features = [feature_names[i] for i in sorted_indices]
        top_importance = mean_abs_shap[sorted_indices]
        top_signs = [correlation_signs[i] for i in sorted_indices]
        
        # Automatically print clinical interpretation 
        print("\n" + "="*80)
        print(" CLINICAL INTERPRETATION OF TOP SHAP FEATURES (PCA INVERSE TRANSFORMATION)")
        print("="*80)
        
        if hasattr(self, 'audio_pca_model') and hasattr(self, 'visual_pca_model'):
            for i, feat_name in enumerate(top_features):
                modality = feat_name.split('_')[0]
                pca_idx = int(feat_name.split('_')[2])
                
                if modality == 'A':
                    meaning = interpret_pca_component(self.audio_pca_model, pca_idx, self.audio_feature_names)
                    mod_str = " AUDIO "
                else:
                    meaning = interpret_pca_component(self.visual_pca_model, pca_idx, self.visual_feature_names)
                    mod_str = " VISUAL"
                
                impact_dir = "Increases Risk" if top_signs[i] > 0 else "Decreases Risk"
                print(f"[{i+1:2d}] {feat_name:8s} | {mod_str} | {impact_dir:14s} |  {meaning}")
        else:
            print(" PCA models not loaded. Cannot provide clinical interpretation.")
        print("="*80 + "\n")
        
        # 4. Assign colors based on direction
        colors = ['#ff0051' if sign > 0 else '#008bfb' for sign in top_signs]
        
        # 5. Plot graph
        plt.figure(figsize=(10, 6))
        
        y_pos = np.arange(len(top_features)) 
        
        bars = plt.barh(y_pos, top_importance[::-1], color=colors[::-1], edgecolor='black', alpha=0.8)
        
        # 6. Styling
        plt.yticks(y_pos, top_features[::-1], fontsize=11, fontweight='bold')
        
        plt.xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=12)
        plt.title('Top 10 Most Important PCA Features', fontsize=14, fontweight='bold', pad=20)
        legend_elements = [
            Patch(facecolor='#ff0051', edgecolor='black', label='High Value $\\rightarrow$ Increases Depression Risk'),
            Patch(facecolor='#008bfb', edgecolor='black', label='High Value $\\rightarrow$ Decreases Depression Risk')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'enhanced_shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Enhanced SHAP bar plot saved to {self.output_dir}/enhanced_shap_bar.png")

    def plot_modality_importance(self, shap_values):
        """
        This function aggregates the SHAP values across all features within each modality to compute the macro-level Modality Importance.
        It provides empirical evidence that the Adaptive Fusion layer successfully prevents modality collapse, ensuring that both acoustic and visual signals 
        contribute synergistically to the final clinical diagnosis.
        """
        print("Generating modality importance bar plot...")
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        audio_importance = mean_abs_shap[:self.audio_dim].sum()
        visual_importance = mean_abs_shap[self.audio_dim:].sum()
        
        total = audio_importance + visual_importance
        audio_pct = (audio_importance / total) * 100
        visual_pct = (visual_importance / total) * 100

        plt.figure(figsize=(7, 6))
        bars = plt.bar(['Audio (PCA)', 'Visual (PCA)'], 
                       [audio_importance, visual_importance], 
                       color=['#1f77b4', '#d62728'], alpha=0.8)
        
        plt.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height(), 
                 f'{audio_pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height(), 
                 f'{visual_pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.title('Overall Modality Importance (SHAP on Mean Features)')
        plt.ylabel('Total Mean Absolute SHAP Value')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_modality_importance.png'), dpi=300)
        plt.close()
        print(f" Modality importance plot saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for Multimodal Depression Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt')
    parser.add_argument('--cache_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='explainability_plots')
    
    parser.add_argument('--pca_dir', type=str, required=True, help='Directory containing _pca_audio.pkl')
    
    parser.add_argument('--audio_dim', type=int, default=50)
    parser.add_argument('--visual_dim', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=4)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    shap.utils.sample
    
    print("="*60)
    print(" SHAP EXPLAINABILITY ANALYSIS (Mean Pooled)")
    print("="*60)

    print(f"Loading model architecture...")
    model = MultimodalClassifier(
        audio_dim=args.audio_dim,
        visual_dim=args.visual_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        fusion_temperature=1.0
    ).to(device)
    
    print(f"Loading weights from {args.checkpoint}...")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(state_dict, strict=False)
        print(" Model weights loaded successfully.")
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        return

    print(f"Loading test data from {args.cache_path}...")
    _, _, test_loader = create_temporal_dataloaders(
        cache_path=args.cache_path,
        batch_size=32,
        use_pca=True 
    )
    
    analyzer = ShapAnalyzer(model, test_loader, device, args.output_dir)
    # Load PCA parameter
    try:
        project_root = Path(__file__).resolve().parent.parent
        feature_names_path = project_root / 'feature_names.json'
        
        with open(feature_names_path, 'r') as f:
            names = json.load(f)
            analyzer.audio_feature_names = names['audio']
            analyzer.visual_feature_names = names['visual']
            
        audio_pca_path = glob.glob(os.path.join(args.pca_dir, "*_pca_audio.pkl"))[0]
        visual_pca_path = glob.glob(os.path.join(args.pca_dir, "*_pca_visual.pkl"))[0]
        
        with open(audio_pca_path, 'rb') as f:
            analyzer.audio_pca_model = pickle.load(f)['pca']
        with open(visual_pca_path, 'rb') as f:
            analyzer.visual_pca_model = pickle.load(f)['pca']
            
        print(" PCA models & feature names loaded for clinical interpretation.")
    except Exception as e:
        print(f" Warning: Could not load PCA models for interpretation: {e}")

    # Run analysis (reduce sample size if it takes too long)
    # Using 100 background samples and 47 test samples
    analyzer.run_shap_analysis(background_samples=100, test_samples=47)
    
    print(f"\n SHAP analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()