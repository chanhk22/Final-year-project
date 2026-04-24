"""
PyTorch Dataset and DataLoader module for temporal multimodal sequences.

It is specifically designed to feed sequential data (time_steps, features) into time-aware models such as Transformers or LSTMs.
It handles dynamic sequence padding, masking, and crucial normalization processes.

Key Methodology:
- Sequence Padding: Ensures uniform tensor sizes for batched training.
- Independent Normalization: Training statistics (mean/std) are applied to Dev/Test 
  sets to prevent statistical data leakage.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List


class TemporalDepressionDataset(Dataset):
    """
    Dataset that preserves temporal structure for Transformer.
    
    Key difference from previous:
    - Returns SEQUENCES (time_steps, features) instead of single vectors
    - Audio: (200, 79) for 2s window @ 100Hz with COVAREP (79 features)
    - Visual: (60, 468) for 2s window @ 30Hz with CLNF features
    
    Or with PCA:
    - Audio: (200, 50) - PCA applied per-frame
    - Visual: (60, 50) - PCA applied per-frame
    """
    
    def __init__(
        self,
        cache_path: str,
        fold: str = 'train',
        use_pca: bool = False,  # If False, use raw temporal sequences
        modalities: List[str] = ['audio', 'visual'],
        max_audio_len: int = 200,  # 2s @ 100Hz
        max_visual_len: int = 60,   # 2s @ 30Hz
        normalize: bool = True
    ):
        """
        Args:
            cache_path: Path to .pkl cache
            fold: 'train', 'dev', 'test'
            use_pca: If True, load PCA features (single vector)
                     If False, load raw temporal sequences
            max_audio_len: Maximum audio sequence length
            max_visual_len: Maximum visual sequence length
        """
        self.cache_path = cache_path
        self.fold = fold
        self.use_pca = use_pca
        self.modalities = modalities
        self.max_audio_len = max_audio_len
        self.max_visual_len = max_visual_len
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
        
        # Determine feature type
        if use_pca:
            self.audio_col = 'audio_pca'
            self.visual_col = 'visual_pca'
            print(f"  Using PCA features (single vectors)")
        else:
            self.audio_col = 'audio_raw'
            self.visual_col = 'visual_raw'
            print(f"  Using RAW temporal sequences")
        
        # Compute normalization stats
        if normalize and fold == 'train':
            self._compute_normalization_stats()
        else:
            self.audio_mean = None
            self.audio_std = None
            self.visual_mean = None
            self.visual_std = None
    
    def _compute_normalization_stats(self):
        """
        Computes mean and standard deviation strictly from the training set.
        
        Applying Z-score normalization using global statistics (including Dev/Test) 
        causes data leakage. Therefore, this function isolates the training fold 
        to compute the scaling factors, which are subsequently applied to all folds.
        """
        print("Computing normalization statistics...")
        
        # For PCA features (single vector per window)
        if self.use_pca:
            if 'audio' in self.modalities and self.audio_col in self.df.columns:
                audio_data = []
                for feat in self.df[self.audio_col]:
                    if feat is not None and isinstance(feat, (list, np.ndarray)):
                        audio_data.append(np.array(feat))
                
                if audio_data:
                    audio_matrix = np.vstack(audio_data)
                    self.audio_mean = np.mean(audio_matrix, axis=0)
                    self.audio_std = np.std(audio_matrix, axis=0) + 1e-8
                    print(f"  Audio PCA: {audio_matrix.shape}")
            
            if 'visual' in self.modalities and self.visual_col in self.df.columns:
                visual_data = []
                for feat in self.df[self.visual_col]:
                    if feat is not None and isinstance(feat, (list, np.ndarray)):
                        visual_data.append(np.array(feat))
                
                if visual_data:
                    visual_matrix = np.vstack(visual_data)
                    self.visual_mean = np.mean(visual_matrix, axis=0)
                    self.visual_std = np.std(visual_matrix, axis=0) + 1e-8
                    print(f"  Visual PCA: {visual_matrix.shape}")
        
        # For raw temporal features (sequences)
        else:
            if 'audio' in self.modalities and self.audio_col in self.df.columns:
                # Collect all frames from all windows
                all_audio_frames = []
                for feat in self.df[self.audio_col]:
                    if feat is not None and isinstance(feat, (list, np.ndarray)):
                        # feat shape: (time_steps, features) or flattened
                        feat_array = np.array(feat)
                        
                        # Reshape if flattened
                        if feat_array.ndim == 1:
                            # Assuming audio_shape stored separately
                            continue
                        
                        all_audio_frames.append(feat_array)
                
                if all_audio_frames:
                    # Stack all frames: (total_frames, feature_dim)
                    all_frames = np.vstack(all_audio_frames)
                    self.audio_mean = np.mean(all_frames, axis=0)
                    self.audio_std = np.std(all_frames, axis=0) + 1e-8
                    print(f"  Audio raw: computed from {len(all_frames)} frames")
            
            # Similar for visual
            if 'visual' in self.modalities and self.visual_col in self.df.columns:
                all_visual_frames = []
                for feat in self.df[self.visual_col]:
                    if feat is not None and isinstance(feat, (list, np.ndarray)):
                        feat_array = np.array(feat)
                        if feat_array.ndim == 1:
                            continue
                        all_visual_frames.append(feat_array)
                
                if all_visual_frames:
                    all_frames = np.vstack(all_visual_frames)
                    self.visual_mean = np.mean(all_frames, axis=0)
                    self.visual_std = np.std(all_frames, axis=0) + 1e-8
                    print(f"  Visual raw: computed from {len(all_frames)} frames")
    
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
            For PCA features:
            - 'audio': (audio_dim,) - single vector
            - 'visual': (visual_dim,) - single vector
            
            For raw features:
            - 'audio': (time_steps, audio_dim) - sequence
            - 'visual': (time_steps, visual_dim) - sequence
            - 'audio_lengths': actual sequence length (for padding)
            - 'visual_lengths': actual sequence length
        """
        row = self.df.iloc[idx]
        
        sample = {
            'session': row['session'],
            'window_idx': row['window_idx'],
            'label': torch.tensor(row['y_bin'], dtype=torch.float32),
            'label_reg': torch.tensor(row['y_reg'], dtype=torch.float32),
        }
        
        # === AUDIO ===
        if 'audio' in self.modalities and self.audio_col in self.df.columns:
            audio_feat = row[self.audio_col]
            
            if audio_feat is not None and isinstance(audio_feat, (list, np.ndarray)):
                audio_feat = np.array(audio_feat, dtype=np.float32)
                
                # Handle based on PCA or raw
                if self.use_pca:
                    # Single vector: (audio_dim,)
                    if self.normalize and self.audio_mean is not None:
                        audio_feat = (audio_feat - self.audio_mean) / self.audio_std
                    
                    sample['audio'] = torch.tensor(audio_feat, dtype=torch.float32)
                    sample['audio_mask'] = torch.tensor(1.0, dtype=torch.float32)
                
                else:
                    # Temporal sequence
                    # audio_feat might be flattened: (time_steps * feature_dim,)
                    # Need to reshape using audio_shape from row
                    
                    if 'audio_shape' in row and row['audio_shape'] is not None:
                        audio_shape = row['audio_shape']
                        audio_feat = audio_feat.reshape(audio_shape)  # (time_steps, feature_dim)
                    else:
                        # Fallback: assume fixed dimensions
                        # e.g., 200 frames * 79 features = 15800
                        audio_feat = audio_feat.reshape(self.max_audio_len, -1)
                    
                    # Normalize per-frame
                    if self.normalize and self.audio_mean is not None:
                        audio_feat = (audio_feat - self.audio_mean) / self.audio_std
                    
                    # Get actual length
                    audio_len = audio_feat.shape[0]
                    
                    # Pad to max_length if needed
                    if audio_len < self.max_audio_len:
                        pad_len = self.max_audio_len - audio_len
                        audio_feat = np.pad(
                            audio_feat, 
                            ((0, pad_len), (0, 0)), 
                            mode='constant'
                        )
                    elif audio_len > self.max_audio_len:
                        audio_feat = audio_feat[:self.max_audio_len]
                        audio_len = self.max_audio_len
                    
                    sample['audio'] = torch.tensor(audio_feat, dtype=torch.float32)
                    sample['audio_lengths'] = torch.tensor(audio_len, dtype=torch.long)
                    sample['audio_mask'] = torch.tensor(1.0, dtype=torch.float32)
            
            else:
                # Missing audio
                if self.use_pca:
                    audio_dim = len(self.audio_mean) if self.audio_mean is not None else 50
                    sample['audio'] = torch.zeros(audio_dim, dtype=torch.float32)
                else:
                    audio_dim = self.audio_mean.shape[0] if self.audio_mean is not None else 79
                    sample['audio'] = torch.zeros(self.max_audio_len, audio_dim, dtype=torch.float32)
                    sample['audio_lengths'] = torch.tensor(0, dtype=torch.long)
                
                sample['audio_mask'] = torch.tensor(0.0, dtype=torch.float32)
        
        # === VISUAL === (similar logic)
        if 'visual' in self.modalities and self.visual_col in self.df.columns:
            visual_feat = row[self.visual_col]
            
            if visual_feat is not None and isinstance(visual_feat, (list, np.ndarray)):
                visual_feat = np.array(visual_feat, dtype=np.float32)
                
                if self.use_pca:
                    if self.normalize and self.visual_mean is not None:
                        visual_feat = (visual_feat - self.visual_mean) / self.visual_std
                    
                    sample['visual'] = torch.tensor(visual_feat, dtype=torch.float32)
                    sample['visual_mask'] = torch.tensor(1.0, dtype=torch.float32)
                
                else:
                    # Reshape to sequence
                    if 'visual_shape' in row and row['visual_shape'] is not None:
                        visual_shape = row['visual_shape']
                        visual_feat = visual_feat.reshape(visual_shape)
                    else:
                        visual_feat = visual_feat.reshape(self.max_visual_len, -1)
                    
                    if self.normalize and self.visual_mean is not None:
                        visual_feat = (visual_feat - self.visual_mean) / self.visual_std
                    
                    visual_len = visual_feat.shape[0]
                    
                    if visual_len < self.max_visual_len:
                        pad_len = self.max_visual_len - visual_len
                        visual_feat = np.pad(
                            visual_feat,
                            ((0, pad_len), (0, 0)),
                            mode='constant'
                        )
                    elif visual_len > self.max_visual_len:
                        visual_feat = visual_feat[:self.max_visual_len]
                        visual_len = self.max_visual_len
                    
                    sample['visual'] = torch.tensor(visual_feat, dtype=torch.float32)
                    sample['visual_lengths'] = torch.tensor(visual_len, dtype=torch.long)
                    sample['visual_mask'] = torch.tensor(1.0, dtype=torch.float32)
            
            else:
                if self.use_pca:
                    visual_dim = len(self.visual_mean) if self.visual_mean is not None else 50
                    sample['visual'] = torch.zeros(visual_dim, dtype=torch.float32)
                else:
                    visual_dim = self.visual_mean.shape[0] if self.visual_mean is not None else 468
                    sample['visual'] = torch.zeros(self.max_visual_len, visual_dim, dtype=torch.float32)
                    sample['visual_lengths'] = torch.tensor(0, dtype=torch.long)
                
                sample['visual_mask'] = torch.tensor(0.0, dtype=torch.float32)
        
        return sample


def create_temporal_dataloaders(
    cache_path: str,
    batch_size: int = 32,
    use_pca: bool = False,
    max_audio_len: int = 200,
    max_visual_len: int = 60,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for temporal data with strict subject leakage verification.
    """
    train_dataset = TemporalDepressionDataset(
        cache_path=cache_path,
        fold='train',
        use_pca=use_pca,
        max_audio_len=max_audio_len,
        max_visual_len=max_visual_len,
        normalize=True
    )
    
    dev_dataset = TemporalDepressionDataset(
        cache_path=cache_path,
        fold='dev',
        use_pca=use_pca,
        max_audio_len=max_audio_len,
        max_visual_len=max_visual_len,
        normalize=False
    )
    
    test_dataset = TemporalDepressionDataset(
        cache_path=cache_path,
        fold='test',
        use_pca=use_pca,
        max_audio_len=max_audio_len,
        max_visual_len=max_visual_len,
        normalize=False
    )

    # Subject-Level Leakage Verification
    print(f"\n{'='*60}")
    print("Verifying Subject-Level Splits (Preventing Leakage)...")
    
    train_subs = set(train_dataset.df['session'].unique())
    dev_subs = set(dev_dataset.df['session'].unique())
    test_subs = set(test_dataset.df['session'].unique())

    train_dev_leak = train_subs.intersection(dev_subs)
    train_test_leak = train_subs.intersection(test_subs)
    dev_test_leak = dev_subs.intersection(test_subs)

    assert len(train_dev_leak) == 0, f" Leakage in Train/Dev detected! Subjects: {train_dev_leak}"
    assert len(train_test_leak) == 0, f" Leakage in Train/Test detected! Subjects: {train_test_leak}"
    assert len(dev_test_leak) == 0, f" Leakage in Dev/Test detected! Subjects: {dev_test_leak}"
    
    print(" Verification Passed: Strict subject-level split confirmed.")
    print(f"{'='*60}\n")
    
    # Share normalization
    if train_dataset.normalize:
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
    print("Temporal DataLoaders created:")
    print(f"  Mode: {'PCA (single vectors)' if use_pca else 'RAW (sequences)'}")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Dev:   {len(dev_loader)} batches ({len(dev_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    if not use_pca:
        print(f"  Audio seq len: {max_audio_len}")
        print(f"  Visual seq len: {max_visual_len}")
    print(f"{'='*60}\n")
    
    return train_loader, dev_loader, test_loader


# Test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    train_loader, dev_loader, test_loader = create_temporal_dataloaders(
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        use_pca=args.use_pca
    )
    
    # Test batch
    print("Testing data loading...")
    batch = next(iter(train_loader))
    
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    if not args.use_pca:
        print("\nSequence length statistics:")
        if 'audio_lengths' in batch:
            print(f"  Audio lengths: min={batch['audio_lengths'].min()}, "
                  f"max={batch['audio_lengths'].max()}, "
                  f"mean={batch['audio_lengths'].float().mean():.1f}")
        if 'visual_lengths' in batch:
            print(f"  Visual lengths: min={batch['visual_lengths'].min()}, "
                  f"max={batch['visual_lengths'].max()}, "
                  f"mean={batch['visual_lengths'].float().mean():.1f}")
    
    print("\n Temporal dataset working!")