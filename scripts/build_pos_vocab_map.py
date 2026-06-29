import os
import json
import torch
from collections import defaultdict
from transformers import BertTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm

def main():
    print("=== Building POS-to-Vocab Mapping ===")
    
    # 1. Load tokenizer and POS mapping
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer)
    
    pos_to_id_path = "data/pos_to_id.json"
    if not os.path.exists(pos_to_id_path):
        raise FileNotFoundError("data/pos_to_id.json not found! Run preprocess.py first.")
        
    with open(pos_to_id_path, "r") as f:
        pos_to_id = json.load(f)
    num_pos_tags = len(pos_to_id)
    print(f"Loaded {num_pos_tags} POS tags.")
    
    # Initialize vocabulary mapping: each POS tag maps to a set of allowed token IDs
    pos_to_tokens = defaultdict(set)
    
    # Always allow all special tokens for all POS tags
    special_ids = tokenizer.all_special_ids
    for pos_id in range(num_pos_tags):
        for spec_id in special_ids:
            pos_to_tokens[pos_id].add(spec_id)
            
    # 2. Load preprocessed datasets
    train_path = "./data/preprocessed_train"
    val_path = "./data/preprocessed_val"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training path {train_path} not found!")
        
    print("Loading preprocessed training dataset...")
    train_dataset = load_from_disk(train_path)
    print(f"Loaded {len(train_dataset)} training examples.")
    
    # 3. Iterate over the dataset and collect token_id -> pos_id co-occurrences
    print("Processing examples to compile token co-occurrences...")
    for idx in tqdm(range(len(train_dataset))):
        example = train_dataset[idx]
        words = example['words']
        pos_labels = example['pos_labels']
        
        # Tokenize words using subword tokenizer
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encoding['input_ids']
        word_ids = encoding.word_ids()
        
        for t_idx, w_id in enumerate(word_ids):
            # Special tokens have w_id = None; they are already handled.
            if w_id is not None and w_id < len(pos_labels):
                token_id = input_ids[t_idx]
                pos_label = pos_labels[w_id]
                if pos_label != -100:
                    pos_to_tokens[pos_label].add(token_id)
                    
    # 4. Handle unobserved vocabulary tokens
    # Find tokens in the 30522 BERT vocabulary that were never observed in the training corpus with any POS tag.
    all_observed_tokens = set()
    for pos_id, tokens in pos_to_tokens.items():
        all_observed_tokens.update(tokens)
        
    unobserved_tokens = set(range(vocab_size)) - all_observed_tokens
    print(f"Total Vocab Size: {vocab_size}")
    print(f"Observed Vocabulary Tokens: {len(all_observed_tokens)}")
    print(f"Unobserved Vocabulary Tokens: {len(unobserved_tokens)} (will be allowed in all POS tags as fallback)")
    
    for pos_id in range(num_pos_tags):
        pos_to_tokens[pos_id].update(unobserved_tokens)
        
    # 5. Save the POS to Vocab mapping
    # Convert sets to sorted lists for clean JSON storage
    pos_to_tokens_dict = {pos_id: sorted(list(tokens)) for pos_id, tokens in pos_to_tokens.items()}
    
    save_path = "data/pos_to_vocab.json"
    with open(save_path, "w") as f:
        json.dump(pos_to_tokens_dict, f)
        
    print(f"Successfully saved POS-to-vocab mapping to {save_path}")

if __name__ == "__main__":
    main()
