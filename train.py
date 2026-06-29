import os
import argparse
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertConfig, get_scheduler
from datasets import load_from_disk
from tqdm import tqdm

from src.models import MultiTaskBERT, LayeredPOSMLMBert
from src.dataset import PretrainDataset, MultiTaskCollator, print_batch_samples

def run_pretraining(args_dict):
    # Parse arguments from dict
    tasks = args_dict.get('tasks', 'mlm,nsp').split(',')
    model_dir = args_dict.get('model_dir', './checkpoints')
    epochs = args_dict.get('epochs', 5)
    batch_size = args_dict.get('batch_size', 32)
    lr = args_dict.get('lr', 5e-5)
    alpha = args_dict.get('alpha', 1.0)
    resume = args_dict.get('resume', False)
    
    print(f"Configuring Pretraining with tasks: {tasks}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, alpha={alpha}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load POS mapping
    if not os.path.exists("data/pos_to_id.json"):
        raise FileNotFoundError("data/pos_to_id.json not found! Run preprocess.py first.")
    with open("data/pos_to_id.json", "r") as f:
        pos_to_id = json.load(f)
    num_pos_tags = len(pos_to_id)
    id_to_pos = {str(v): k for k, v in pos_to_id.items()}
    
    # 2. Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # 3. Load preprocessed datasets
    if not os.path.exists("./data/preprocessed_train") or not os.path.exists("./data/preprocessed_val"):
        raise FileNotFoundError("Preprocessed data directories not found! Run preprocess.py first.")
        
    print("Loading preprocessed dataset from disk...")
    train_hf = load_from_disk("./data/preprocessed_train")
    val_hf = load_from_disk("./data/preprocessed_val")
    
    # 4. Create PyTorch datasets and loaders
    # NSP is active only if 'nsp' is in tasks
    nsp_active = 'nsp' in tasks
    train_dataset = PretrainDataset(train_hf, nsp_active=nsp_active)
    val_dataset = PretrainDataset(val_hf, nsp_active=nsp_active)
    
    collator = MultiTaskCollator(tokenizer, mlm_active='mlm' in tasks)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    # 5. Initialize Model
    # We use a custom configuration matching the strict-small track (BERT-Mini scale)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        classifier_dropout=0.1
    )
    
    model_type = args_dict.get('model_type', 'MultiTaskBERT')
    mask_type = args_dict.get('mask_type', 'soft')
    
    if model_type == 'LayeredPOSMLMBert':
        print(f"Initializing LayeredPOSMLMBert with mask_type={mask_type}...")
        model = LayeredPOSMLMBert(config, num_pos_tags=num_pos_tags)
    else:
        print(f"Initializing MultiTaskBERT...")
        model = MultiTaskBERT(config, num_pos_tags=num_pos_tags)
    
    # 6. Device, Optimizer, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model placed on device: {device}")
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 7. Checkpoint Resuming
    latest_checkpoint_path = os.path.join(model_dir, "checkpoint_latest.pt")
    if resume and os.path.exists(latest_checkpoint_path):
        print(f"Resuming pretraining from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from Epoch {start_epoch}")
        
    # 8. Training loop
    first_batch = True
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        # Use progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Send batch to device
            batch_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Print sample visualization from the first batch
            if first_batch:
                print_batch_samples(batch, tokenizer, id_to_pos)
                first_batch = False
                
            # Forward pass
            if model_type == 'LayeredPOSMLMBert':
                outputs = model(**batch_inputs, tasks=tasks, alpha=alpha, mask_type=mask_type)
            else:
                outputs = model(**batch_inputs, tasks=tasks, alpha=alpha)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            metric_str = f"Loss: {loss.item():.4f}"
            for k in ['mlm_loss', 'pos_loss', 'nsp_loss']:
                if k in outputs:
                    metric_str += f" | {k}: {outputs[k]:.4f}"
            pbar.set_postfix_str(metric_str)
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        print(f"Running evaluation for Epoch {epoch+1}...")
        with torch.no_grad():
            for batch in val_loader:
                batch_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                if model_type == 'LayeredPOSMLMBert':
                    outputs = model(**batch_inputs, tasks=tasks, alpha=alpha, mask_type=mask_type)
                else:
                    outputs = model(**batch_inputs, tasks=tasks, alpha=alpha)
                loss = outputs['loss']
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 9. Save Checkpoint (latest & best)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        # Save latest
        torch.save(checkpoint_state, latest_checkpoint_path)
        print(f"Saved latest checkpoint to {latest_checkpoint_path}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_state['best_val_loss'] = best_val_loss
            best_path = os.path.join(model_dir, "checkpoint_best.pt")
            torch.save(checkpoint_state, best_path)
            print(f"New best model validation loss: {best_val_loss:.4f}. Saved checkpoint to {best_path}")
            
    print("Pretraining finished successfully!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain BERT with Multi-task objectives")
    parser.add_argument("--tasks", type=str, default="mlm,nsp", help="Tasks to pretrain: comma-separated combination of mlm, nsp, pos")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for POS loss")
    parser.add_argument("--model_type", type=str, default="MultiTaskBERT", choices=["MultiTaskBERT", "LayeredPOSMLMBert"], help="Model architecture")
    parser.add_argument("--mask_type", type=str, default="soft", choices=["soft", "hard", "gold"], help="Layered model masking strategy")
    parser.add_argument("--resume", action="store_true", help="Resume pretraining from latest checkpoint if available")
    
    args = parser.parse_args()
    run_pretraining(vars(args))
