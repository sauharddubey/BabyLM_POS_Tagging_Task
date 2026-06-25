import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class BERTMLMHead(nn.Module):
    """
    Standard MLM prediction head (Linear -> Activation -> LayerNorm -> Linear).
    """
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

    def forward(self, hidden_states):
        return self.predictions(hidden_states)

class MultiTaskBERT(nn.Module):
    """
    Multi-task BERT model supporting MLM, NSP, and POS prediction tasks.
    POS prediction is done by pooling subword hidden states into a single word representation.
    """
    def __init__(self, config: BertConfig, num_pos_tags: int):
        super().__init__()
        self.config = config
        self.num_pos_tags = num_pos_tags
        
        # 1. Base BERT encoder
        self.bert = BertModel(config)
        
        # 2. Masked Language Modeling Head
        self.mlm_head = BERTMLMHead(config)
        
        # 3. Next Sentence Prediction Head
        self.nsp_head = nn.Linear(config.hidden_size, 2)
        
        # 4. Part-of-Speech Prediction Head
        self.pos_classifier = nn.Linear(config.hidden_size, num_pos_tags)

        # Tie the weights of the final MLM projection layer with the word embeddings
        self.mlm_head.predictions[3].weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,                 # MLM target labels [batch_size, seq_len]
        next_sentence_label=None,    # NSP target labels [batch_size]
        word_ids=None,               # Map from subwords to word index [batch_size, seq_len], -1 for special tokens
        pos_labels=None,             # Word-level POS labels [batch_size, max_words]
        tasks=None,                  # List of tasks to compute: ['mlm', 'nsp', 'pos']
        alpha=1.0                    # Weight for POS loss relative to MLM
    ):
        if tasks is None:
            tasks = ['mlm', 'nsp', 'pos']
            
        # Get hidden states from BERT base model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        pooler_output = outputs[1]    # [batch_size, hidden_size]
        
        losses = {}
        total_loss = 0.0
        outputs_dict = {}
        
        # --- 1. Masked Language Modeling (MLM) Task ---
        if 'mlm' in tasks:
            mlm_logits = self.mlm_head(sequence_output)
            outputs_dict['mlm_logits'] = mlm_logits
            if labels is not None:
                # Masked LM loss
                mlm_loss = F.cross_entropy(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                losses['mlm_loss'] = mlm_loss
                
        # --- 2. Next Sentence Prediction (NSP) Task ---
        if 'nsp' in tasks:
            nsp_logits = self.nsp_head(pooler_output)
            outputs_dict['nsp_logits'] = nsp_logits
            if next_sentence_label is not None:
                nsp_loss = F.cross_entropy(nsp_logits.view(-1, 2), next_sentence_label.view(-1))
                losses['nsp_loss'] = nsp_loss

        # --- 3. Part-of-Speech (POS) Prediction Task ---
        if 'pos' in tasks:
            if word_ids is not None and pos_labels is not None:
                # Find the maximum word ID in the batch to define the pooling dimension
                valid_word_ids = word_ids[word_ids >= 0]
                if valid_word_ids.numel() > 0:
                    max_words = int(valid_word_ids.max().item()) + 1
                else:
                    max_words = 0

                if max_words > 0:
                    # Construct vectorized pooling matrix P: [batch_size, max_words, seq_len]
                    word_indices = torch.arange(max_words, device=word_ids.device).view(1, max_words, 1)
                    mask = (word_ids.unsqueeze(1) == word_indices).float()  # [batch_size, max_words, seq_len]
                    
                    subword_counts = mask.sum(dim=-1, keepdim=True)  # [batch_size, max_words, 1]
                    subword_counts = torch.clamp(subword_counts, min=1.0)
                    
                    P = mask / subword_counts  # [batch_size, max_words, seq_len]
                    
                    # Pool subword states into word representations
                    pooled_word_reprs = torch.bmm(P, sequence_output)  # [batch_size, max_words, hidden_size]
                    
                    # Align token-level pos_labels (shape [batch_size, seq_len]) to word-level aligned_pos_labels (shape [batch_size, max_words])
                    index_tensor = (word_ids.unsqueeze(1) == word_indices)
                    first_indices = index_tensor.double().argmax(dim=-1)  # [batch_size, max_words]
                    
                    aligned_pos_labels = torch.gather(pos_labels, dim=1, index=first_indices)  # [batch_size, max_words]
                    word_present = index_tensor.any(dim=-1)  # [batch_size, max_words]
                    aligned_pos_labels = aligned_pos_labels.masked_fill(~word_present, -100)

                    # Predict POS tags
                    pos_logits = self.pos_classifier(pooled_word_reprs)  # [batch_size, max_words, num_pos_tags]
                    
                    # Compute classification loss
                    pos_loss = F.cross_entropy(
                        pos_logits.view(-1, self.num_pos_tags),
                        aligned_pos_labels.reshape(-1),
                        ignore_index=-100
                    )
                    
                    losses['pos_loss'] = pos_loss
                    outputs_dict['pos_logits'] = pos_logits
                else:
                    losses['pos_loss'] = torch.tensor(0.0, device=input_ids.device)
            else:
                losses['pos_loss'] = torch.tensor(0.0, device=input_ids.device)

        # --- Aggregate Losses ---
        # Different combinations have different aggregation functions:
        active_losses = [losses[k] for k in losses]
        if len(active_losses) > 0:
            if set(tasks) == {'mlm', 'nsp', 'pos'}:
                # MLM_POS_NSPBERT: Loss is MLM + NSP + alpha * POS (consistent sum aggregation)
                mlm_val = losses.get('mlm_loss', torch.tensor(0.0, device=input_ids.device))
                nsp_val = losses.get('nsp_loss', torch.tensor(0.0, device=input_ids.device))
                pos_val = losses.get('pos_loss', torch.tensor(0.0, device=input_ids.device))
                total_loss = mlm_val + nsp_val + alpha * pos_val
            elif set(tasks) == {'mlm', 'pos'}:
                # MLMPOSBert: Loss is MLM + alpha * POS
                mlm_val = losses.get('mlm_loss', torch.tensor(0.0, device=input_ids.device))
                pos_val = losses.get('pos_loss', torch.tensor(0.0, device=input_ids.device))
                total_loss = mlm_val + alpha * pos_val
            elif 'pos' in tasks:
                # Other tasks containing POS (e.g. NSPPOSBERT, POSBERT) apply alpha to pos_loss
                non_pos_losses = [losses[k] for k in losses if k != 'pos_loss']
                pos_val = losses.get('pos_loss', torch.tensor(0.0, device=input_ids.device))
                total_loss = sum(non_pos_losses) + alpha * pos_val
            else:
                # Simple sum for tasks without POS (e.g. BASEBERT)
                total_loss = sum(active_losses)
                
            outputs_dict['loss'] = total_loss
            
        # Add individual losses to outputs dict
        for k, v in losses.items():
            outputs_dict[k] = v.item()
            
        return outputs_dict
