import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import EncoderDecoderConfig, EncoderDecoderModel
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import corpus_bleu

from src.models import MultiTaskBERT
from src.dataset import GLUEDataset, collate_glue, TranslationDataset, collate_seq2seq

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Perplexity Evaluation ---
def evaluate_perplexity(model, tokenizer, dataset, desc="Perplexity"):
    """
    Computes pseudo-perplexity on a dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for example in tqdm(dataset, desc=desc):
            text = example.get('text', '')
            if not text or not text.strip():
                continue
                
            enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(device)
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            
            # Skip short sequences
            if input_ids.size(1) <= 2:
                continue
                
            # Compute pseudo-perplexity by masking tokens one by one
            # PLL(S) = sum_t log P(x_t | S_{\setminus t})
            special_mask = tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
            non_special_indices = [idx for idx, is_spec in enumerate(special_mask) if not is_spec]
            
            if not non_special_indices:
                continue
                
            n_tokens = len(non_special_indices)
            masked_inputs = input_ids.repeat(n_tokens, 1)
            batch_attention_mask = attention_mask.repeat(n_tokens, 1)
            
            for i, t in enumerate(non_special_indices):
                masked_inputs[i, t] = tokenizer.mask_token_id
                
            outputs = model(
                input_ids=masked_inputs,
                attention_mask=batch_attention_mask,
                tasks=['mlm']
            )
            logits = outputs['mlm_logits']  # [n_tokens, seq_len, vocab_size]
            
            row_indices = torch.arange(n_tokens, device=device)
            col_indices = torch.tensor(non_special_indices, device=device)
            target_tokens = torch.tensor([input_ids[0, t].item() for t in non_special_indices], device=device)
            
            masked_logits = logits[row_indices, col_indices]
            losses = F.cross_entropy(masked_logits, target_tokens, reduction='none')
            
            total_loss += losses.sum().item()
            total_tokens += n_tokens
                
    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

# --- 2. BLiMP Zero-Shot Evaluation ---
def calculate_sentence_pll(model, tokenizer, sentence):
    """
    Calculates Pseudo-Log-Likelihood (PLL) for a sentence.
    """
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    
    seq_len = input_ids.size(1)
    if seq_len <= 2:
        return -9999.0
        
    special_mask = tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
    non_special_indices = [idx for idx, is_spec in enumerate(special_mask) if not is_spec]
    
    if not non_special_indices:
        return -9999.0
        
    n_tokens = len(non_special_indices)
    masked_inputs = input_ids.repeat(n_tokens, 1)
    batch_attention_mask = attention_mask.repeat(n_tokens, 1)
    
    for i, t in enumerate(non_special_indices):
        masked_inputs[i, t] = tokenizer.mask_token_id
        
    outputs = model(
        input_ids=masked_inputs,
        attention_mask=batch_attention_mask,
        tasks=['mlm']
    )
    logits = outputs['mlm_logits']  # [n_tokens, seq_len, vocab_size]
    
    row_indices = torch.arange(n_tokens, device=device)
    col_indices = torch.tensor(non_special_indices, device=device)
    target_tokens = torch.tensor([input_ids[0, t].item() for t in non_special_indices], device=device)
    
    masked_logits = logits[row_indices, col_indices]
    log_probs = F.log_softmax(masked_logits, dim=-1)
    target_log_probs = log_probs[row_indices, target_tokens]
    
    return target_log_probs.sum().item()

def evaluate_blimp(model, tokenizer, dataset, desc="BLiMP", max_samples=100):
    """
    Evaluates zero-shot grammatical accuracy on a BLiMP dataset.
    """
    model.eval()
    correct = 0
    total = 0
    
    # We sample a subset for fast validation
    samples = list(dataset)[:max_samples]
    
    with torch.no_grad():
        for example in tqdm(samples, desc=desc):
            sent_good = example['sentence_good']
            sent_bad = example['sentence_bad']
            
            pll_good = calculate_sentence_pll(model, tokenizer, sent_good)
            pll_bad = calculate_sentence_pll(model, tokenizer, sent_bad)
            
            if pll_good > pll_bad:
                correct += 1
            total += 1
            
    if total == 0:
        return 0.0
    return (correct / total) * 100.0

# --- 3. GLUE Downstream Fine-Tuning ---


def fine_tune_glue(bert_config, encoder_state_dict, tokenizer, task_name, train_ds, val_ds, epochs=3, batch_size=32):
    """
    Fine-tunes the BERT model on a sequence classification task.
    """
    print(f"\n--- Fine-tuning on GLUE {task_name.upper()} ---")
    
    num_labels = 2
    bert_config.num_labels = num_labels
    classifier_model = BertForSequenceClassification(bert_config)
    
    # Load pretrained encoder weights
    # Strip the 'bert.' prefix
    cleaned_state_dict = {k[5:]: v for k, v in encoder_state_dict.items() if k.startswith("bert.")}
    classifier_model.bert.load_state_dict(cleaned_state_dict, strict=False)
    classifier_model.to(device)
    
    # Pre-process datasets
    train_dataset = GLUEDataset(train_ds, tokenizer, task_name)
    val_dataset = GLUEDataset(val_ds, tokenizer, task_name)
    
    # Optional: Subsample SST-2 to keep training fast
    if task_name == "sst2" and len(train_dataset) > 10000:
        indices = np.random.choice(len(train_dataset), 10000, replace=False)
        train_dataset.features = [train_dataset.features[i] for i in indices]
        print(f"Subsampled SST-2 training dataset to {len(train_dataset)} samples.")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_glue(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_glue(b, tokenizer))
    
    optimizer = AdamW(classifier_model.parameters(), lr=2e-5)
    
    # Training Loop
    for epoch in range(epochs):
        classifier_model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier_model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    # Evaluation Loop
    classifier_model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier_model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = batch['labels'].cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    # Compute metrics
    metrics = {}
    if task_name == "cola":
        metrics['mcc'] = matthews_corrcoef(all_targets, all_preds)
    elif task_name == "mrpc":
        metrics['acc'] = accuracy_score(all_targets, all_preds)
        metrics['f1'] = f1_score(all_targets, all_preds)
    elif task_name == "sst2":
        metrics['acc'] = accuracy_score(all_targets, all_preds)
        
    print(f"GLUE {task_name.upper()} Results: {metrics}")
    return metrics

# --- 4. BLEU Translation Evaluation ---


def fine_tune_bleu(bert_config, encoder_state_dict, tokenizer, train_ds, test_ds, epochs=3, batch_size=32, max_eval_samples=200):
    """
    Fine-tunes an EncoderDecoder model on Multi30k and calculates BLEU score.
    """
    print("\n--- Fine-tuning Seq2Seq Model on Multi30k (BLEU) ---")
    
    decoder_tokenizer = tokenizer
    
    # Setup Encoder-Decoder Config
    decoder_config = BertConfig(
        vocab_size=len(decoder_tokenizer),
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        max_position_embeddings=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        pad_token_id=decoder_tokenizer.pad_token_id,
        is_decoder=True,
        add_cross_attention=True
    )
    
    enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(bert_config, decoder_config)
    # Set padding/bos/eos token ids
    enc_dec_config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    enc_dec_config.pad_token_id = decoder_tokenizer.pad_token_id
    enc_dec_config.eos_token_id = decoder_tokenizer.sep_token_id
    
    seq2seq_model = EncoderDecoderModel(config=enc_dec_config)
    
    # Load pretrained BERT encoder
    cleaned_state_dict = {k[5:]: v for k, v in encoder_state_dict.items() if k.startswith("bert.")}
    seq2seq_model.encoder.load_state_dict(cleaned_state_dict, strict=False)
    seq2seq_model.to(device)
    
    # Setup datasets
    train_dataset = TranslationDataset(train_ds)
    test_dataset = TranslationDataset(test_ds)
    
    # Optional: limit training/testing for speed
    if len(train_dataset) > 10000:
        indices = np.random.choice(len(train_dataset), 10000, replace=False).tolist()
        train_dataset.dataset = train_dataset.dataset.select(indices)
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_seq2seq(b, tokenizer, decoder_tokenizer)
    )
    
    optimizer = AdamW(seq2seq_model.parameters(), lr=5e-5)
    
    # Train
    for epoch in range(epochs):
        seq2seq_model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = seq2seq_model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    # Evaluate BLEU on test set
    seq2seq_model.eval()
    ref_sentences = []
    gen_sentences = []
    
    # Subset testing to keep translation generation fast
    eval_subset = list(test_dataset)[:max_eval_samples]
    
    with torch.no_grad():
        for item in tqdm(eval_subset, desc="Generating translations"):
            en_text = item['en']
            de_text = item['de']
            
            inputs = tokenizer(en_text, return_tensors="pt").to(device)
            generated_ids = seq2seq_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=64,
                num_beams=2,
                decoder_start_token_id=decoder_tokenizer.cls_token_id,
                eos_token_id=decoder_tokenizer.sep_token_id
            )
            
            gen_text = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            ref_sentences.append(de_text)
            gen_sentences.append(gen_text)
            
    # Compute NLTK corpus BLEU
    references = [[[tok for tok in ref.lower().split()]] for ref in ref_sentences]
    hypotheses = [[tok for tok in hyp.lower().split()] for hyp in gen_sentences]
    
    bleu_score = corpus_bleu(references, hypotheses) * 100.0
    print(f"Multi30k BLEU Score: {bleu_score:.4f}")
    return bleu_score


# --- Main Evaluation Driver ---
def run_evaluation(args):
    print(f"\n======================================")
    print(f"Evaluating checkpoint: {args.checkpoint_path}")
    print(f"======================================\n")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")
        
    # Load checkpoints
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    encoder_state_dict = checkpoint['model_state_dict']
    
    # Setup tokenizers and configs
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    with open("data/pos_to_id.json", "r") as f:
        pos_to_id = json.load(f)
    num_pos_tags = len(pos_to_id)
    
    # Scaled down configuration (BERT-Mini scale)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Initialize MultiTaskBERT model
    model = MultiTaskBERT(config, num_pos_tags=num_pos_tags)
    # Load state dict
    model.load_state_dict(encoder_state_dict)
    model.to(device)
    
    eval_results = {}
    active_eval_tasks = args.tasks.split(",")
    
    # A. Perplexity Evaluation
    if "perplexity" in active_eval_tasks:
        print("\n--- Evaluating Perplexity ---")
        # 1. WikiText-2 (held-out)
        try:
            wt_dataset = load_from_disk("./data/eval_data/wikitext-2")['validation']
            # Limit wikitext size to 500 samples for evaluation speed
            wt_subset = wt_dataset.select(range(min(500, len(wt_dataset))))
            wt_ppl = evaluate_perplexity(model, tokenizer, wt_subset, desc="WikiText-2 PPL")
            eval_results['wikitext2_perplexity'] = wt_ppl
            print(f"WikiText-2 Perplexity: {wt_ppl:.4f}")
        except Exception as e:
            print(f"Error evaluating WikiText-2 Perplexity: {e}")
            
        # 2. Local BabyLM validation split
        try:
            babylm_val = load_from_disk("./data/preprocessed_val")
            # Limit to 500 samples
            babylm_val_subset = babylm_val.select(range(min(500, len(babylm_val))))
            val_ppl = evaluate_perplexity(model, tokenizer, babylm_val_subset, desc="BabyLM Val PPL")
            eval_results['babylm_perplexity'] = val_ppl
            print(f"BabyLM Validation Perplexity: {val_ppl:.4f}")
        except Exception as e:
            print(f"Error evaluating BabyLM Perplexity: {e}")
            
    # B. BLiMP Zero-Shot Evaluation
    if "blimp" in active_eval_tasks:
        print("\n--- Evaluating BLiMP Zero-shot ---")
        for subset in ["regular_plural_subject_verb_agreement_1", "anaphor_gender_agreement"]:
            try:
                bl_dataset = load_from_disk(f"./data/eval_data/blimp_{subset}")['train']
                accuracy = evaluate_blimp(model, tokenizer, bl_dataset, desc=f"BLiMP-{subset}", max_samples=100)
                eval_results[f"blimp_{subset}_accuracy"] = accuracy
                print(f"BLiMP {subset} Accuracy: {accuracy:.2f}%")
            except Exception as e:
                print(f"Error evaluating BLiMP {subset}: {e}")
                
    # C. GLUE Downstream Fine-Tuning
    if "glue" in active_eval_tasks:
        # MRPC
        try:
            mrpc_dataset = load_from_disk("./data/eval_data/glue_mrpc")
            mrpc_metrics = fine_tune_glue(config, encoder_state_dict, tokenizer, "mrpc", mrpc_dataset['train'], mrpc_dataset['validation'], epochs=3)
            eval_results['glue_mrpc_accuracy'] = mrpc_metrics['acc']
            eval_results['glue_mrpc_f1'] = mrpc_metrics['f1']
        except Exception as e:
            print(f"Error evaluating GLUE MRPC: {e}")
            
        # CoLA
        try:
            cola_dataset = load_from_disk("./data/eval_data/glue_cola")
            cola_metrics = fine_tune_glue(config, encoder_state_dict, tokenizer, "cola", cola_dataset['train'], cola_dataset['validation'], epochs=3)
            eval_results['glue_cola_mcc'] = cola_metrics['mcc']
        except Exception as e:
            print(f"Error evaluating GLUE CoLA: {e}")
            
        # SST-2
        try:
            sst_dataset = load_from_disk("./data/eval_data/glue_sst2")
            sst_metrics = fine_tune_glue(config, encoder_state_dict, tokenizer, "sst2", sst_dataset['train'], sst_dataset['validation'], epochs=3)
            eval_results['glue_sst2_accuracy'] = sst_metrics['acc']
        except Exception as e:
            print(f"Error evaluating GLUE SST-2: {e}")
            
    # D. BLEU Evaluation
    if "bleu" in active_eval_tasks:
        try:
            m30_dataset = load_from_disk("./data/eval_data/multi30k")
            bleu_score = fine_tune_bleu(config, encoder_state_dict, tokenizer, m30_dataset['train'], m30_dataset['test'], epochs=3, max_eval_samples=200)
            eval_results['multi30k_bleu'] = bleu_score
        except Exception as e:
            print(f"Error evaluating BLEU Multi30k: {e}")
            
    # 5. Output results as JSON file in checkpoint folder
    out_file = args.checkpoint_path.replace(".pt", "_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nSaved evaluation results to {out_file}")
    
    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pretrained BERT Checkpoints")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file (e.g. checkpoint_best.pt)")
    parser.add_argument("--tasks", type=str, default="perplexity,blimp,glue,bleu", help="Evaluation tasks: perplexity,blimp,glue,bleu")
    args = parser.parse_args()
    run_evaluation(args)
