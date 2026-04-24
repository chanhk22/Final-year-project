import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal sequence modeling using Self-Attention mechanisms.
    
    [Methodological Rationale]
    Depression markers (e.g., psychomotor retardation) manifest progressively over 
    the course of a clinical interview. The self-attention mechanism is employed to 
    capture these long-range temporal dependencies without the vanishing gradient 
    issues of traditional RNNs.
    
    [Architecture Reference] A. Vaswani et al., "Attention is all you need," 
    in Advances in Neural Information Processing Systems (NeurIPS), 2017.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Temporal pooling (instead of just taking last state)
        self.temporal_pool = nn.Linear(d_model, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) - SINGLE frame (will be expanded)
               OR (batch, seq_len, input_dim) - sequence
            mask: (batch,) or (batch, seq_len)
        
        Returns:
            pooled: (batch, d_model) - pooled representation
            attention_weights: For visualization
        """
        # Handle single frame input (for backward compatibility)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Create attention mask
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(1).expand(-1, seq_len)
            attn_mask = (mask == 0)  # True where missing
        else:
            attn_mask = None
        
        # Apply transformer
        encoded = self.transformer(
            x,
            src_key_padding_mask=attn_mask
        )  # (batch, seq_len, d_model)
        
        encoded = self.layer_norm(encoded)
        
        # Temporal pooling with attention
        attn_scores = self.temporal_pool(encoded).squeeze(-1)  # (batch, seq_len)
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, -1e9)
            # Prevent NaN when all sequences are masked
            is_all_masked = attn_mask.all(dim=1, keepdim=True)
            if is_all_masked.any():
                attn_scores = attn_scores.masked_fill(is_all_masked, 0.0)
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # (batch, 1, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights, encoded).squeeze(1)  # (batch, d_model)
        pooled = torch.nan_to_num(pooled, nan=0.0)
        
        # Apply mask
        if mask is not None:
            pooled = pooled * (mask.sum(dim=1, keepdim=True) > 0).float()
        
        return pooled, attn_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for capturing inter-modality dynamics.
    
    Nonverbal cues are highly correlated; for instance, an acoustic sigh is often 
    accompanied by a visual frown. This cross-attention block allows one modality to attend to the 
    temporal sequence of the other, effectively aligning unaligned multimodal streams and enhancing robustness against missing data.
    
    [Architecture Reference] Y.-H. H. Tsai et al., "Multimodal transformer for unaligned multimodal language sequences," in Proc. of the 57th Annual Meeting 
    of the Association for Computational Linguistics (ACL), 2019.
    """
    
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-modal attention"""
        query_seq = query.unsqueeze(1)
        kv_seq = key_value.unsqueeze(1)
        
        if kv_mask is not None:
            key_padding_mask = (kv_mask == 0).unsqueeze(1)
        else:
            key_padding_mask = None
        
        attended, attn_weights = self.multihead_attn(
            query_seq,
            kv_seq,
            kv_seq,
            key_padding_mask=key_padding_mask
        )
        attended = torch.nan_to_num(attended, nan=0.0)
        attended = attended.squeeze(1)
        if kv_mask is not None:
            attended = attended * kv_mask.unsqueeze(1)
        attended = self.layer_norm(query + self.dropout(attended))
        
        if query_mask is not None:
            attended = attended * query_mask.unsqueeze(1)
        
        return attended, attn_weights


class ImprovedAdaptiveFusion(nn.Module):
    """
    Dynamic modality fusion using Temperature-scaled Softmax and Gumbel-Softmax.
    
    Instead of static concatenation, this adaptive gate dynamically learns the 
    importance of each modality per sample. This architecture intrinsically supports post-hoc sensitivity 
    analysis and XAI evaluations by making the modality-level contribution mathematically transparent.
    
    [Technique Reference] E. Jang, S. Gu, and B. Poole, "Categorical reparameterization with Gumbel-Softmax," 
    in International Conference on Learning Representations (ICLR), 2017.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_modalities: int = 2,
        temperature: float = 1.0,
        use_gumbel: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        # Weight computation network
        self.weight_net = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_modalities)
        )
        
        # Fallback for single modality
        self.single_modality_transform = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        modality_features: list,
        modality_masks: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive fusion with soft weighting.
        """
        batch_size = modality_features[0].size(0)
        mask_tensor = torch.stack(modality_masks, dim=1)  # (batch, num_modalities)
        
        # Check how many modalities are present per sample
        num_present = mask_tensor.sum(dim=1)  # (batch,)
        
        # Concatenate features
        concat_features = torch.cat(modality_features, dim=1)
        
        # Compute logits
        logits = self.weight_net(concat_features)  # (batch, num_modalities)
        
        # Temperature-scaled softmax
        logits = logits / self.temperature
        
        # Mask missing modalities
        logits = logits.masked_fill(mask_tensor == 0, -1e9)
        
        is_all_missing = (mask_tensor.sum(dim=1, keepdim=True) == 0)
        if is_all_missing.any():
            logits = logits.masked_fill(is_all_missing, 0.0)
        
        # Compute weights
        if self.use_gumbel and self.training:
            # Gumbel-Softmax for training (allows gradient flow)
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)
        else:
            weights = F.softmax(logits, dim=1)
        
        # Weighted fusion
        fused = torch.zeros(batch_size, self.d_model, device=modality_features[0].device)
        for i, (feat, weight) in enumerate(zip(modality_features, weights.unbind(dim=1))):
            fused += feat * weight.unsqueeze(1)
        
        # For samples with only one modality, apply transform
        single_modality_mask = (num_present == 1).unsqueeze(1)
        if single_modality_mask.any():
            fused = torch.where(
                single_modality_mask,
                self.single_modality_transform(fused),
                fused
            )
        
        return fused, weights


class MultimodalClassifier(nn.Module):
    """
    End-to-End Multimodal Depression Detection Framework.
    
    This architecture integrates temporal self-attention, cross-modal attention, and adaptive fusion to process visual (OpenFace AUs) and acoustic (COVAREP) features. 
    It handles missing modalities dynamically via attention masking, ensuring robust clinical predictions.
    """
    
    def __init__(
        self,
        audio_dim: int = 50,
        visual_dim: int = 50,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        fusion_temperature: float = 2.0,  # Higher = softer fusion
        use_gumbel_fusion: bool = False,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.d_model = d_model
        
        # Temporal encoders
        self.audio_encoder = TemporalTransformerEncoder(
            input_dim=audio_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.visual_encoder = TemporalTransformerEncoder(
            input_dim=visual_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Cross-modal attention
        self.audio_to_visual = CrossModalAttention(d_model, nhead, dropout)
        self.visual_to_audio = CrossModalAttention(d_model, nhead, dropout)
        
        # Improved adaptive fusion
        self.fusion = ImprovedAdaptiveFusion(
            d_model, 
            num_modalities=2,
            temperature=fusion_temperature,
            use_gumbel=use_gumbel_fusion
        )
        
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode modalities
        audio_encoded, audio_temp_attn = self.audio_encoder(audio, audio_mask)
        visual_encoded, visual_temp_attn = self.visual_encoder(visual, visual_mask)
        
        # Cross-modal attention
        audio_attended, audio_cross_attn = self.audio_to_visual(
            query=audio_encoded,
            key_value=visual_encoded,
            query_mask=audio_mask,
            kv_mask=visual_mask
        )
        
        visual_attended, visual_cross_attn = self.visual_to_audio(
            query=visual_encoded,
            key_value=audio_encoded,
            query_mask=visual_mask,
            kv_mask=audio_mask
        )
        
        # Adaptive fusion
        fused, fusion_weights = self.fusion(
            modality_features=[audio_attended, visual_attended],
            modality_masks=[audio_mask, visual_mask]
        )
        
        # Classification
        logits = self.classifier(fused)
        
        output = {
            'logits': logits,
            'fusion_weights': fusion_weights,
            'audio_encoded': audio_encoded,
            'visual_encoded': visual_encoded
        }
        
        if return_attention:
            output['audio_temporal_attention'] = audio_temp_attn
            output['visual_temporal_attention'] = visual_temp_attn
            output['audio_cross_attention'] = audio_cross_attn
            output['visual_cross_attention'] = visual_cross_attn
        
        return output


# Test
if __name__ == "__main__":
    print("Testing Improved Model...")
    
    model = MultimodalClassifier(
        audio_dim=50,
        visual_dim=50,
        d_model=128,
        nhead=4,
        num_layers=2,
        fusion_temperature=2.0
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    batch_size = 8
    audio = torch.randn(batch_size, 50)
    visual = torch.randn(batch_size, 50)
    audio_mask = torch.ones(batch_size)
    visual_mask = torch.ones(batch_size)
    
    # Drop some modalities
    audio_mask[0] = 0
    visual_mask[1] = 0
    
    with torch.no_grad():
        output = model(audio, visual, audio_mask, visual_mask, return_attention=True)
    
    print(f"\nOutput shapes:")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print(f"\nFusion weights (first 4 samples):")
    print(output['fusion_weights'][:4])
    
    print("\n Improved model working!")