import os
import json
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from dataset.dataset import create_dataloaders, get_subject_level_predictions
from models.model import MultimodalClassifier

class EarlyStopping:
    """
    Early stopping regularization to prevent overfitting.
    
    Given the small sample size of clinical datasets like DAIC-WOZ, models are highly prone to memorizing the training set. 
    This class monitors the validation (dev) loss/metric and halts training when generalization stops improving, 
    ensuring the model remains robust on unseen test data.
    """
    
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.labels = []
        self.sessions = []
        self.losses = []
    
    def update(self, preds, labels, sessions, loss):
        self.predictions.extend(preds.detach().cpu().numpy().tolist())
        self.labels.extend(labels.detach().cpu().numpy().tolist())
        self.sessions.extend(sessions)
        self.losses.append(loss)
    
    def compute_metrics(self, threshold=0.5):
        """Compute window-level and subject-level metrics"""
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Window-level metrics
        preds_binary = (preds > threshold).astype(int)
        
        window_metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(labels, preds_binary),
            'precision': precision_score(labels, preds_binary, zero_division=0),
            'recall': recall_score(labels, preds_binary, zero_division=0),
            'f1': f1_score(labels, preds_binary, zero_division=0),
            'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
        }
        
        # Subject-level metrics
        subject_preds, subject_labels, unique_sessions = get_subject_level_predictions(
            preds, labels, self.sessions
        )
        subject_preds_binary = (subject_preds > threshold).astype(int)
        
        subject_metrics = {
            'accuracy': accuracy_score(subject_labels, subject_preds_binary),
            'precision': precision_score(subject_labels, subject_preds_binary, zero_division=0),
            'recall': recall_score(subject_labels, subject_preds_binary, zero_division=0),
            'f1': f1_score(subject_labels, subject_preds_binary, zero_division=0),
            'auc': roc_auc_score(subject_labels, subject_preds) if len(np.unique(subject_labels)) > 1 else 0.0,
            # Return original data 
            'y_true': subject_labels.tolist(),
            'y_pred': subject_preds.tolist()
        }
        
        return window_metrics, subject_metrics


class ImprovedTrainer:
    """
    Advanced training pipeline tailored for multimodal clinical time-series.
    
    1. AdamW Optimizer: Decouples weight decay from gradient updates, providing 
       better generalization for transformer-based architectures than standard Adam.
       (Reference: I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," ICLR, 2019)
       
    2. Cosine Annealing with Warm Restarts: Helps the model escape local minima by periodically resetting the learning rate, 
       which is crucial for complex non-convex loss landscapes in multimodal fusion.
       
    3. Fusion Entropy Regularization (fusion_entropy_weight): Prevents the model 
       from collapsing into relying on a single dominant modality (e.g., only audio). 
       It forces the network to distribute attention across both audio and visual streams, maximizing complementary information extraction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        dev_loader,
        test_loader,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        max_epochs: int = 100,
        patience: int = 15,
        grad_clip: float = 1.0,
        # Fusion regularization
        fusion_entropy_weight: float = 0.1,
        # Top-K aggregation
        use_topk_aggregation: bool = False,
        topk_ratio: float = 0.3,
        save_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        experiment_name: str = 'exp'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        
        # parameters
        self.fusion_entropy_weight = fusion_entropy_weight
        self.use_topk_aggregation = use_topk_aggregation
        self.topk_ratio = topk_ratio
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Tracking
        self.save_dir = os.path.join(save_dir, experiment_name)
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.best_dev_f1 = 0.0
        self.history = defaultdict(list)
        
        print(f"\n{'='*60}")
        print("Improved Training Configuration")
        print(f"{'='*60}")
        print(f"Fusion entropy weight: {fusion_entropy_weight}")
        print(f"Top-K aggregation: {use_topk_aggregation} (ratio={topk_ratio})")
        print(f"Gradient Clipping: {grad_clip}")
        print(f"{'='*60}\n")
    
    
    def compute_fusion_entropy_loss(self, fusion_weights):
        eps = 1e-8
        entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + eps), dim=1)
        max_entropy = np.log(2)
        entropy_loss = -entropy.mean() / max_entropy
        return entropy_loss
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch with improvements"""
        self.model.train()
        tracker = MetricsTracker()
        
        total_cls_loss = 0.0
        total_entropy_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            audio_mask = batch['audio_mask'].to(self.device)
            visual_mask = batch['visual_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            sessions = batch['session']
            
            output = self.model(audio, visual, audio_mask, visual_mask)
            logits = output['logits'].squeeze(1)
            fusion_weights = output['fusion_weights']
            
            cls_loss = self.criterion(logits, labels)
            entropy_loss = self.compute_fusion_entropy_loss(fusion_weights)
            
            loss = cls_loss + self.fusion_entropy_weight * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            probs = torch.sigmoid(logits)
            tracker.update(probs, labels, sessions, loss.item())
            
            total_cls_loss += cls_loss.item()
            total_entropy_loss += entropy_loss.item()
            
            pbar.set_postfix({
                'cls': cls_loss.item(),
                'ent': entropy_loss.item()
            })
        
        window_metrics, subject_metrics = tracker.compute_metrics()
        self.scheduler.step()
        
        return {
            'window': window_metrics,
            'subject': subject_metrics,
            'lr': self.optimizer.param_groups[0]['lr'],
            'cls_loss': total_cls_loss / len(self.train_loader),
            'entropy_loss': total_entropy_loss / len(self.train_loader)
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader, split_name='dev') -> dict:
        self.model.eval()
        if self.use_topk_aggregation and split_name == 'test':
            return self.evaluate_with_topk(data_loader, split_name)
        else:
            return self.evaluate_standard(data_loader, split_name)
    
    @torch.no_grad()
    def evaluate_standard(self, data_loader, split_name='dev') -> dict:
        tracker = MetricsTracker()
        pbar = tqdm(data_loader, desc=f'Evaluating [{split_name}]')
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            audio_mask = batch['audio_mask'].to(self.device)
            visual_mask = batch['visual_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            sessions = batch['session']
            
            output = self.model(audio, visual, audio_mask, visual_mask)
            logits = output['logits'].squeeze(1)
            
            loss = self.criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            tracker.update(probs, labels, sessions, loss.item())
        
        window_metrics, subject_metrics = tracker.compute_metrics()
        
        return {
            'window': window_metrics,
            'subject': subject_metrics
        }
    
    @torch.no_grad()
    def evaluate_with_topk(self, data_loader, split_name='test') -> dict:
        """
        Depressive symptoms do not manifest uniformly; they often appear in short, intense bursts (e.g., a momentary sigh or a fleeting micro-expression 
        of sadness). Instead of averaging all windows equally—which might dilute these sparse signals—Top-K aggregation focuses the final subject-level 
        decision strictly on the 'K' most discriminative and confident windows.
        """
        session_windows = defaultdict(lambda: {'preds': [], 'label': None})
        pbar = tqdm(data_loader, desc=f'Evaluating [{split_name}] with Top-K')
        
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            audio_mask = batch['audio_mask'].to(self.device)
            visual_mask = batch['visual_mask'].to(self.device)
            labels = batch['label']
            sessions = batch['session']
            
            output = self.model(audio, visual, audio_mask, visual_mask)
            logits = output['logits'].squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, sess in enumerate(sessions):
                session_windows[sess]['preds'].append(probs[i])
                session_windows[sess]['label'] = labels[i].item()
        
        subject_preds = []
        subject_labels = []
        
        for sess, data in session_windows.items():
            preds = np.array(data['preds'])
            label = data['label']
            if len(preds) == 0:
                continue
            
            k = max(1, int(len(preds) * self.topk_ratio))
            confidence = np.abs(preds - 0.5)
            topk_indices = np.argsort(confidence)[-k:]
            topk_preds = preds[topk_indices]
            
            subject_pred = np.mean(topk_preds)
            subject_preds.append(subject_pred)
            subject_labels.append(label)
        
        subject_preds = np.array(subject_preds)
        subject_labels = np.array(subject_labels)
        subject_preds_binary = (subject_preds > 0.5).astype(int)
        
        subject_metrics = {
            'accuracy': accuracy_score(subject_labels, subject_preds_binary),
            'precision': precision_score(subject_labels, subject_preds_binary, zero_division=0),
            'recall': recall_score(subject_labels, subject_preds_binary, zero_division=0),
            'f1': f1_score(subject_labels, subject_preds_binary, zero_division=0),
            'auc': roc_auc_score(subject_labels, subject_preds) if len(np.unique(subject_labels)) > 1 else 0.0,
            # Return original data
            'y_true': subject_labels.tolist(),
            'y_pred': subject_preds.tolist()
        }
        
        return {
            'window': {'loss': 0.0, 'f1': 0.0, 'auc': 0.0},
            'subject': subject_metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dev_f1': self.best_dev_f1,
            'history': dict(self.history)
        }
        
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f'  → Saved best model (F1: {self.best_dev_f1:.4f})')
    
    def log_metrics(self, epoch: int, train_metrics: dict, dev_metrics: dict):
        self.writer.add_scalars('Loss', {'train': train_metrics['window']['loss'], 'dev': dev_metrics['window']['loss']}, epoch)
        self.writer.add_scalar('Loss/cls_train', train_metrics['cls_loss'], epoch)
        self.writer.add_scalar('Loss/entropy_train', train_metrics['entropy_loss'], epoch)
        self.writer.add_scalars('Window/F1', {'train': train_metrics['window']['f1'], 'dev': dev_metrics['window']['f1']}, epoch)
        self.writer.add_scalars('Subject/F1', {'train': train_metrics['subject']['f1'], 'dev': dev_metrics['subject']['f1']}, epoch)
        self.writer.add_scalars('Subject/AUC', {'train': train_metrics['subject']['auc'], 'dev': dev_metrics['subject']['auc']}, epoch)
        self.writer.add_scalar('LearningRate', train_metrics['lr'], epoch)
        self.history['train_subject_f1'].append(train_metrics['subject']['f1'])
        self.history['dev_subject_f1'].append(dev_metrics['subject']['f1'])
    
    def train(self):
        print(f"\n{'='*60}")
        print("Starting Improved Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            dev_metrics = self.evaluate(self.dev_loader, 'dev')
            self.log_metrics(epoch, train_metrics, dev_metrics)
            
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            print(f"  Train - Loss: {train_metrics['window']['loss']:.4f}, Sub-F1: {train_metrics['subject']['f1']:.4f}, Ent: {train_metrics['entropy_loss']:.4f}")
            print(f"  Dev   - Loss: {dev_metrics['window']['loss']:.4f}, Sub-F1: {dev_metrics['subject']['f1']:.4f}")
            
            dev_f1 = dev_metrics['subject']['f1']
            is_best = dev_f1 > self.best_dev_f1
            if is_best:
                self.best_dev_f1 = dev_f1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.early_stopping(dev_f1):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        
        best_checkpoint = torch.load(os.path.join(self.save_dir, 'best.pt'), weights_only=False)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        print(f"\nTest Results:")
        print(f"  Subject-level:")
        print(f"    F1:  {test_metrics['subject']['f1']:.4f}")
        print(f"    AUC: {test_metrics['subject']['auc']:.4f}")
        print(f"    Recall: {test_metrics['subject']['recall']:.4f}")
        print(f"    Precision: {test_metrics['subject']['precision']:.4f}")
        
        if 'y_true' in test_metrics['subject'] and 'y_pred' in test_metrics['subject']:
            df_preds = pd.DataFrame({
                'true_label': test_metrics['subject']['y_true'],
                'pred_prob': test_metrics['subject']['y_pred']
            })
            csv_path = os.path.join(self.save_dir, 'test_predictions.csv')
            df_preds.to_csv(csv_path, index=False)
            print(f" Saved predictions to {csv_path} for Bootstrap CI calculation.")

            del test_metrics['subject']['y_true']
            del test_metrics['subject']['y_pred']
        
        results = {
            'best_epoch': best_checkpoint['epoch'],
            'best_dev_f1': self.best_dev_f1,
            'test_metrics': test_metrics
        }
        
        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        
        return results


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_pca', action='store_true', default=True)
    parser.add_argument('--no_pca', action='store_false', dest='use_pca')
    
    parser.add_argument('--audio_dim', type=int, default=50)
    parser.add_argument('--visual_dim', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fusion_temperature', type=float, default=2.0)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--fusion_entropy_weight', type=float, default=0.1)
    parser.add_argument('--use_topk', action='store_true')
    parser.add_argument('--topk_ratio', type=float, default=0.3)
    
    parser.add_argument('--save_dir', default='checkpoints_improved')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--experiment_name', default='exp_improved')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not args.use_pca:
        print("\n" + "="*60)
        print(" WARNING: Using RAW features (no PCA)")
        print("="*60)
        args.audio_dim = 79
        args.visual_dim = 393
        if args.batch_size > 16:
            args.batch_size = 16
        if args.d_model < 256:
            args.d_model = 256
        print(f" Audio dim: {args.audio_dim}")
        print(f" Visual dim: {args.visual_dim}")
        print(f" Batch size: {args.batch_size}")
        print(f" d_model: {args.d_model}")
        print("="*60 + "\n")
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading data...")
    train_loader, dev_loader, test_loader = create_dataloaders(
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        use_pca=args.use_pca,
        modalities=['audio', 'visual'],
        normalize=True,
        num_workers=args.num_workers
    )

    if args.pos_weight == 1.0:
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['label'].numpy().tolist())
        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count
        args.pos_weight = neg_count / pos_count
        print(f"Calculated pos_weight: {args.pos_weight:.4f}")
    
    print("\nCreating model...")
    model = MultimodalClassifier(
        audio_dim=args.audio_dim,
        visual_dim=args.visual_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        fusion_temperature=args.fusion_temperature
    )
    
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        fusion_entropy_weight=args.fusion_entropy_weight,
        use_topk_aggregation=args.use_topk,
        topk_ratio=args.topk_ratio,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    results = trainer.train()
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()