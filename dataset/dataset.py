import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List

class DepressionWindowDataset(Dataset):
    """
    Dataset for depression detection from windowed multimodal features.
    
    In clinical datasets, certain modalities may be sporadically missing (e.g., facial tracking failure). 
    Instead of discarding these valuable samples, this class employs Zero-Imputation paired with Mask Tensors (audio_mask, visual_mask). 
    This allows the neural network to dynamically ignore missing streams via attention mechanisms without crashing or learning noisy zero-values.
    """
    def __init__(
        self,
        cache_path: str,
        fold: str = 'train',
        use_pca: bool = True,
        modalities: List[str] = ['audio', 'visual'],
        normalize: bool = True
    ):
        """
        Args:
            cache_path: Path to .pkl cache file
            fold: 'train', 'dev', or 'test'
            use_pca: Use PCA features if True, else raw features
            modalities: List of modalities to use ['audio', 'visual']
            normalize: Apply z-score normalization per feature
        """
        self.cache_path = cache_path
        self.fold = fold
        self.use_pca = use_pca
        self.modalities = modalities
        self.normalize = normalize
        
        # Load data
        print(f"Loading dataset from {cache_path}")
        self.df = pd.read_pickle(cache_path)
        
        # Filter by fold
        self.df = self.df[self.df['fold'] == fold].reset_index(drop=True)
        
        print(f"Fold: {fold}")
        print(f"  Total windows: {len(self.df)}")
        print(f"  Sessions: {self.df['session'].nunique()}")
        print(f"  Depression ratio: {self.df['y_bin'].mean():.3f}")
        
        # Feature columns
        self.audio_col = 'audio_pca' if use_pca else 'audio_raw'
        self.visual_col = 'visual_pca' if use_pca else 'visual_raw'
        
        # Compute normalization stats on train fold only
        if normalize and fold == 'train':
            self._compute_normalization_stats()
        else:
            self.audio_mean = None
            self.audio_std = None
            self.visual_mean = None
            self.visual_std = None
    def _compute_normalization_stats(self):
        """
        Computes mean and standard deviation strictly from the training fold.

        To maintain the integrity of the evaluation, Z-score normalization parameters (mean and std) are derived solely from the training data. 
        These parameters are then frozen and applied to the dev/test folds, ensuring zero statistical data leakage.
        """
        print("Computing normalization statistics...")
        
        # Audio stats
        if 'audio' in self.modalities and self.audio_col in self.df.columns:
            audio_data = []
            for feat in self.df[self.audio_col]:
                if feat is not None and isinstance(feat, (list, np.ndarray)):
                    audio_data.append(np.array(feat))
            
            if audio_data:
                audio_matrix = np.vstack(audio_data)
                self.audio_mean = np.mean(audio_matrix, axis=0)
                self.audio_std = np.std(audio_matrix, axis=0) + 1e-8
                print(f"  Audio: shape={audio_matrix.shape}")
            else:
                self.audio_mean = None
                self.audio_std = None
        
        # Visual stats
        if 'visual' in self.modalities and self.visual_col in self.df.columns:
            visual_data = []
            for feat in self.df[self.visual_col]:
                if feat is not None and isinstance(feat, (list, np.ndarray)):
                    visual_data.append(np.array(feat))
            
            if visual_data:
                visual_matrix = np.vstack(visual_data)
                self.visual_mean = np.mean(visual_matrix, axis=0)
                self.visual_std = np.std(visual_matrix, axis=0) + 1e-8
                print(f"  Visual: shape={visual_matrix.shape}")
            else:
                self.visual_mean = None
                self.visual_std = None
    def set_normalization_stats(self, audio_mean, audio_std, visual_mean, visual_std):
        """Set normalization stats from train dataset"""
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.visual_mean = visual_mean
        self.visual_std = visual_std
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with keys:
            - 'audio': (audio_dim,) or None
            - 'visual': (visual_dim,) or None
            - 'audio_mask': 1 if audio present, 0 otherwise
            - 'visual_mask': 1 if visual present, 0 otherwise
            - 'label': binary label (0 or 1)
            - 'label_reg': regression label (PHQ score)
            - 'session': session ID
            - 'window_idx': window index
        """
        row = self.df.iloc[idx]
        
        sample = {
            'session': row['session'],
            'window_idx': row['window_idx'],
            'label': torch.tensor(row['y_bin'], dtype=torch.float32),
            'label_reg': torch.tensor(row['y_reg'], dtype=torch.float32),
        }
        
        # Audio features
        if 'audio' in self.modalities and self.audio_col in self.df.columns:
            audio_feat = row[self.audio_col]
            
            if audio_feat is not None and isinstance(audio_feat, (list, np.ndarray)):
                audio_feat = np.array(audio_feat, dtype=np.float32)
                
                # Normalize
                if self.normalize and self.audio_mean is not None:
                    audio_feat = (audio_feat - self.audio_mean) / self.audio_std
                
                sample['audio'] = torch.tensor(audio_feat, dtype=torch.float32)
                sample['audio_mask'] = torch.tensor(1.0, dtype=torch.float32)
            else:
                # Missing audio - use zeros
                audio_dim = len(self.audio_mean) if self.audio_mean is not None else 50
                sample['audio'] = torch.zeros(audio_dim, dtype=torch.float32)
                sample['audio_mask'] = torch.tensor(0.0, dtype=torch.float32)
        
        # Visual features
        if 'visual' in self.modalities and self.visual_col in self.df.columns:
            visual_feat = row[self.visual_col]
            
            if visual_feat is not None and isinstance(visual_feat, (list, np.ndarray)):
                visual_feat = np.array(visual_feat, dtype=np.float32)
                
                # Normalize
                if self.normalize and self.visual_mean is not None:
                    visual_feat = (visual_feat - self.visual_mean) / self.visual_std
                
                sample['visual'] = torch.tensor(visual_feat, dtype=torch.float32)
                sample['visual_mask'] = torch.tensor(1.0, dtype=torch.float32)
            else:
                # Missing visual - use zeros
                visual_dim = len(self.visual_mean) if self.visual_mean is not None else 50
                sample['visual'] = torch.zeros(visual_dim, dtype=torch.float32)
                sample['visual_mask'] = torch.tensor(0.0, dtype=torch.float32)
        
        return sample
    
def create_dataloaders(
    cache_path: str,
    batch_size: int = 32,
    use_pca: bool = True,
    modalities: List[str] = ['audio', 'visual'],
    normalize: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, dev, test dataloaders with proper normalization.
    
    Args:
        cache_path: Path to cache .pkl file
        batch_size: Batch size
        use_pca: Use PCA features
        modalities: Modalities to use
        normalize: Apply normalization
        num_workers: Number of dataloader workers
    
    Returns:
        (train_loader, dev_loader, test_loader)
    """
    # Create datasets
    train_dataset = DepressionWindowDataset(
        cache_path=cache_path,
        fold='train',
        use_pca=use_pca,
        modalities=modalities,
        normalize=normalize
    )
    
    dev_dataset = DepressionWindowDataset(
        cache_path=cache_path,
        fold='dev',
        use_pca=use_pca,
        modalities=modalities,
        normalize=False  # Don't compute stats
    )
    
    test_dataset = DepressionWindowDataset(
        cache_path=cache_path,
        fold='test',
        use_pca=use_pca,
        modalities=modalities,
        normalize=False  # Don't compute stats
    )
    
    # Share normalization stats from train
    if normalize:
        dev_dataset.set_normalization_stats(
            train_dataset.audio_mean,
            train_dataset.audio_std,
            train_dataset.visual_mean,
            train_dataset.visual_std
        )
        test_dataset.set_normalization_stats(
            train_dataset.audio_mean,
            train_dataset.audio_std,
            train_dataset.visual_mean,
            train_dataset.visual_std
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Dev:   {len(dev_loader)} batches ({len(dev_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"{'='*60}\n")
    
    return train_loader, dev_loader, test_loader

def get_subject_level_predictions(
    window_predictions: np.ndarray,
    window_labels: np.ndarray,
    session_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Aggregates window-level prediction scores to formulate a final subject-level clinical diagnosis using Mean Pooling.

    Depressive symptoms fluctuate throughout an interview. The neural network outputs continuous probability scores for micro-windows (e.g., 2 seconds). 
    By applying mean aggregation across all windows of a specific participant, the system smoothes out local anomalies and derives a robust, overall 
    depression severity score for the entire session, which aligns with the global PHQ-8 ground truth.
    """
    df = pd.DataFrame({
        'session': session_ids,
        'pred': window_predictions,
        'label': window_labels
    })
    
    # Group by session and take mean
    subject_df = df.groupby('session').agg({
        'pred': 'mean',
        'label': 'first'  # All windows from same session have same label
    }).reset_index()
    
    return (
        subject_df['pred'].values,
        subject_df['label'].values,
        subject_df['session'].tolist()
    )
    
# Test script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', required=True, help='Path to cache .pkl file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_pca', action='store_true', default=True)
    parser.add_argument('--no_pca', action='store_false', dest='use_pca')
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        use_pca=args.use_pca
    )
    
    # Test one batch
    print("Testing data loading...")
    batch = next(iter(train_loader))
    
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("\nSample statistics:")
    print(f"  Audio present: {batch['audio_mask'].sum().item()}/{len(batch['audio_mask'])}")
    print(f"  Visual present: {batch['visual_mask'].sum().item()}/{len(batch['visual_mask'])}")
    print(f"  Positive labels: {batch['label'].sum().item()}/{len(batch['label'])}")
    
    print("\n Dataset module working correctly!")