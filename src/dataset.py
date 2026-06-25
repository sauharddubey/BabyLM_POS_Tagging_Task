import random
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Dataset wrapping the preprocessed BabyLM corpus.
    Pairs sentences for NSP dynamically if the task is active.
    """
    def __init__(self, hf_dataset, nsp_active=True):
        self.dataset = hf_dataset
        self.nsp_active = nsp_active

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        words = example['words']
        pos_labels = example['pos_labels']
        
        # If NSP is active, construct sentence pairs (50% positive, 50% negative)
        if self.nsp_active:
            is_next = random.random() > 0.5
            if is_next:
                # Pair with the consecutive sentence in the corpus
                next_idx = idx + 1 if idx + 1 < len(self.dataset) else idx - 1
                next_example = self.dataset[next_idx]
                words_b = next_example['words']
                pos_b = next_example['pos_labels']
                next_label = 0
            else:
                # Pick a random sentence B
                rand_idx = random.randint(0, len(self.dataset) - 1)
                while rand_idx == idx:
                    rand_idx = random.randint(0, len(self.dataset) - 1)
                rand_example = self.dataset[rand_idx]
                words_b = rand_example['words']
                pos_b = rand_example['pos_labels']
                next_label = 1
            words_a = words
            pos_a = pos_labels
        else:
            # No NSP: single sentence/paragraph
            words_a = words
            pos_a = pos_labels
            words_b = None
            pos_b = None
            next_label = 0  # Dummy label
            
        return {
            "words_a": words_a,
            "pos_a": pos_a,
            "words_b": words_b,
            "pos_b": pos_b,
            "next_label": next_label
        }

class MultiTaskCollator:
    """
    Data collator that tokenizes sentence pairs, aligns POS tags using word_ids, 
    and applies MLM masking on the fly.
    """
    def __init__(self, tokenizer, mlm_active=True, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_active = mlm_active
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        # 1. Tokenize batch text lists
        batch_text_a = [item['words_a'] for item in batch]
        batch_text_b = [item['words_b'] for item in batch]
        
        # Check if text_b is active
        has_text_b = batch_text_b[0] is not None
        
        if has_text_b:
            encoding = self.tokenizer(
                text=batch_text_a,
                text_pair=batch_text_b,
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                text=batch_text_a,
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt"
            )
            
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids))
        
        # 2. Build MLM Labels and apply masking
        labels = input_ids.clone()
        
        if self.mlm_active:
            # Mask out special tokens ([PAD]=0, [CLS]=101, [SEP]=102, [MASK]=103)
            # Create a probability matrix
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK] = 103)
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]

            # The remaining 10% of the time, we keep the masked input tokens unchanged
        else:
            labels.fill_(-100)
            
        # 3. Align Word IDs and POS Labels
        # Since NSP resets word_ids for text_pair, we must offset text_pair word_ids by length of text_a
        batch_size, seq_len = input_ids.shape
        aligned_pos_labels = []
        aligned_word_ids = []
        
        for b in range(batch_size):
            word_ids = encoding.word_ids(b)
            item = batch[b]
            pos_a = item['pos_a']
            pos_b = item['pos_b']
            
            len_a = len(item['words_a'])
            
            example_pos_labels = []
            example_word_ids = []
            
            for t in range(seq_len):
                w_id = word_ids[t]
                t_type = token_type_ids[b, t].item()
                
                if w_id is None:
                    example_word_ids.append(-1)
                    example_pos_labels.append(-100)
                elif t_type == 0:
                    example_word_ids.append(w_id)
                    # Get label for text_a
                    label = pos_a[w_id] if w_id < len(pos_a) else -100
                    example_pos_labels.append(label)
                else:  # text_pair
                    # Offset the word ID by the number of words in text_a
                    example_word_ids.append(w_id + len_a)
                    # Get label for text_b
                    label = pos_b[w_id] if w_id < len(pos_b) else -100
                    example_pos_labels.append(label)
                    
            aligned_word_ids.append(example_word_ids)
            aligned_pos_labels.append(example_pos_labels)
            
        aligned_word_ids = torch.tensor(aligned_word_ids, dtype=torch.long)
        aligned_pos_labels = torch.tensor(aligned_pos_labels, dtype=torch.long)
        
        next_sentence_label = torch.tensor([item['next_label'] for item in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "next_sentence_label": next_sentence_label,
            "word_ids": aligned_word_ids,
            "pos_labels": aligned_pos_labels
        }

def print_batch_samples(batch, tokenizer, id_to_pos, num_samples=2):
    """
    Prints input samples in detail in the terminal.
    """
    print(f"\n--- Printing Batch Samples (First {num_samples}) ---")
    input_ids = batch['input_ids']
    labels = batch['labels']
    word_ids = batch['word_ids']
    pos_labels = batch['pos_labels']
    token_type_ids = batch['token_type_ids']
    next_sentence_label = batch['next_sentence_label']
    
    for i in range(min(num_samples, input_ids.size(0))):
        print(f"\n[Sample {i+1}] NSP Label: {next_sentence_label[i].item()}")
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        
        # Build alignment columns
        cols = []
        for t_idx, tok in enumerate(tokens):
            lbl_id = labels[i, t_idx].item()
            lbl_tok = tokenizer.convert_ids_to_tokens([lbl_id])[0] if lbl_id != -100 else "IGNORE"
            w_id = word_ids[i, t_idx].item()
            pos_id = pos_labels[i, t_idx].item()
            pos_tag = id_to_pos[str(pos_id)] if str(pos_id) in id_to_pos else "IGNORE"
            t_type = token_type_ids[i, t_idx].item()
            
            cols.append(f"  Token {t_idx:2d}: {tok:15s} | Segment: {t_type} | WordID: {w_id:2d} | MLM Target: {lbl_tok:12s} | POS Label: {pos_tag}")
            
        # Print first 25 tokens to keep output size readable
        for line in cols[:25]:
            print(line)
        if len(cols) > 25:
            print(f"  ... and {len(cols)-25} more tokens.")
    print("---------------------------------------------------\n")

# --- Downstream Evaluation Datasets & Collators ---
class GLUEDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, task_name):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.features = []
        self._preprocess()

    def _preprocess(self):
        for item in self.dataset:
            # Handle different task keys
            if self.task_name == "cola":
                text1 = item['sentence']
                text2 = None
                label = item['label']
            elif self.task_name == "mrpc":
                text1 = item['sentence1']
                text2 = item['sentence2']
                label = item['label']
            elif self.task_name == "sst2":
                text1 = item['sentence']
                text2 = None
                label = item['label']
            else:
                continue
                
            if label == -1: # Skip invalid inputs
                continue
                
            self.features.append((text1, text2, label))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        text1, text2, label = self.features[idx]
        return {"text1": text1, "text2": text2, "label": label}

def collate_glue(batch, tokenizer):
    texts1 = [item['text1'] for item in batch]
    texts2 = [item['text2'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    if texts2[0] is not None:
        enc = tokenizer(texts1, texts2, padding=True, truncation=True, max_length=128, return_tensors="pt")
    else:
        enc = tokenizer(texts1, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
    enc['labels'] = labels
    return enc

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"en": item['en'], "de": item['de']}

def collate_seq2seq(batch, tokenizer, decoder_tokenizer):
    en_texts = [item['en'] for item in batch]
    de_texts = [item['de'] for item in batch]
    
    # Encode inputs (English)
    inputs = tokenizer(en_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    
    # Encode labels (German)
    with decoder_tokenizer.as_target_tokenizer():
        labels = decoder_tokenizer(de_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")['input_ids']
        
    # Replace padding token id with -100 to ignore in loss
    labels[labels == decoder_tokenizer.pad_token_id] = -100
    
    inputs['labels'] = labels
    return inputs
