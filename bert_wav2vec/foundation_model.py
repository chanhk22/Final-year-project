import torch
import torch.nn as nn
from transformers import BertModel, Wav2Vec2Model
from peft import LoraConfig, get_peft_model

class FoundationMultimodalClassifier(nn.Module):
    """
    Multimodal Foundation Model Framework integrating BERT and Wav2Vec 2.0.
    
    [Parameter-Efficient Fine-Tuning (PEFT)]
    Fully fine-tuning large foundation models like BERT on small clinical datasets (e.g., DAIC-WOZ) inevitably leads to severe overfitting and catastrophic forgetting. 
    To overcome this, implement Low-Rank Adaptation (LoRA). By freezing the pre-trained model weights and injecting trainable rank decomposition matrices 
    into the Transformer layers, achieve domain-specific linguistic adaptation (psychiatric interview context) while maintaining profound generalization capabilities.
    
    [Architecture References]
    1. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL, 2019.
    2. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022.
    """
    def __init__(self, 
                 audio_input_dim=768, 
                 text_model_name='bert-base-uncased', 
                 audio_model_name="facebook/wav2vec2-base-960h",
                 d_model=128, 
                 num_layers=2, 
                 nhead=4, 
                 dropout=0.3, 
                 use_lora=True): 
        super().__init__()
        
        # 1. Text Encoder (BERT)
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(768, d_model)
        """
        [LoRA Configuration]
        Target Modules: "query" and "value" projection matrices within BERT's self-attention mechanism are adapted. 
        This optimally balances the trade-off between training stability (low trainable parameters) and the model's ability 
        to shift its semantic focus toward depressive linguistic markers.
        """
        if use_lora:
            print(" Applying LoRA to BERT (Fine-tuning mode on!)")
            peft_config = LoraConfig(
                inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                target_modules=["query", "value"] # Attacb LoRA to BERT Attention
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()
        else:
            # Freeze 
            for param in self.bert.parameters():
                param.requires_grad = False

        # 3. Audio Projector
        self.audio_proj = nn.Linear(audio_input_dim, d_model)
        
        # 4. Multimodal Transformer (Cross-Attention alternative)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=dropout, batch_first=True
        )
        self.audio_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Fusion & Classifier
        self.fusion_weight = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask, audio_features, audio_mask):
        # BERT (LoRA adapted)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # PEFT model output can be different
        if hasattr(bert_out, 'pooler_output'):
            text_emb = bert_out.pooler_output
        else:
            text_emb = bert_out[1] # Pooler output usually at index 1
            
        text_emb = self.text_proj(text_emb)
        
        # Audio
        audio_emb = self.audio_proj(audio_features)
        key_mask = (1.0 - audio_mask).bool()
        audio_out = self.audio_transformer(audio_emb, src_key_padding_mask=key_mask)
        
        # Mean Pooling
        mask_expanded = audio_mask.unsqueeze(-1)
        audio_sum = (audio_out * mask_expanded).sum(dim=1)
        audio_len = mask_expanded.sum(dim=1).clamp(min=1e-9)
        audio_emb = audio_sum / audio_len
        
        # Fusion
        combined = torch.cat([text_emb, audio_emb], dim=1)
        weights = self.fusion_weight(combined)
        fused_emb = (weights[:, 0:1] * text_emb) + (weights[:, 1:2] * audio_emb)
        
        logits = self.classifier(fused_emb)
        
        return {
            'logits': logits,
            'fusion_weights': weights
        }