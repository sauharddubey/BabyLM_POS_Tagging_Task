import os
import json
import nltk
from datasets import load_dataset, DatasetDict

def main():
    print("=== Start Dataset Preprocessing ===")
    
    # 1. Setup NLTK data directory inside workspace
    nltk_data_dir = os.path.abspath("./data/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    print(f"Downloading NLTK resources to {nltk_data_dir}...")
    try:
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
        
    try:
        nltk.data.find('tokenizers/punkt_tab', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('punkt_tab', download_dir=nltk_data_dir)
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng', paths=[nltk_data_dir])
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
        except Exception:
            pass

    # 2. Load and process pretraining dataset
    print("Loading pretraining dataset: BabyLM-community/BabyLM-2026-Strict-Small...")
    raw_dataset = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
    
    print("Running word-level tokenization and POS tagging on training data...")
    def pos_tag_fn(example):
        text = example.get('text', '')
        if not text or not text.strip():
            return {"words": [], "pos_tags": []}
        
        words = nltk.word_tokenize(text)
        try:
            pos_tuples = nltk.pos_tag(words)
        except Exception:
            pos_tuples = [(w, "UNK") for w in words]
            
        pos_tags = [tag for _, tag in pos_tuples]
        return {"words": words, "pos_tags": pos_tags}

    processed = raw_dataset.map(pos_tag_fn, num_proc=4, desc="Tokenizing and POS Tagging")
    processed = processed.filter(lambda x: len(x['words']) > 0, desc="Filtering empty examples")
    
    print("Building POS tag vocabulary...")
    unique_tags = set()
    for item in processed['train']:
        unique_tags.update(item['pos_tags'])
        
    unique_tags = sorted(list(unique_tags))
    pos_to_id = {tag: idx for idx, tag in enumerate(unique_tags)}
    print(f"Found {len(pos_to_id)} unique POS tags: {list(pos_to_id.keys())}")
    
    with open("data/pos_to_id.json", "w") as f:
        json.dump(pos_to_id, f, indent=4)
    print("Saved data/pos_to_id.json")
    
    def map_tags_to_ids(example):
        example['pos_labels'] = [pos_to_id[tag] for tag in example['pos_tags']]
        return example
        
    mapped_dataset = processed.map(map_tags_to_ids, desc="Mapping tag strings to IDs")
    
    print("Splitting pretraining dataset into 95% train and 5% validation sequentially...")
    split_index = int(len(mapped_dataset['train']) * 0.95)
    train_dataset = mapped_dataset['train'].select(range(split_index))
    val_dataset = mapped_dataset['train'].select(range(split_index, len(mapped_dataset['train'])))
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Print sample to terminal
    sample = dataset_dict['train'][0]
    print("\n--- Preprocessed Pretraining Sample 1 ---")
    paired = [f"{w}/{t}" for w, t in zip(sample['words'], sample['pos_tags'])]
    print(" ".join(paired[:30]))
    print(f"POS Labels: {sample['pos_labels'][:30]}")
    print("-------------------------------------------\n")
    
    train_path = "./data/preprocessed_train"
    val_path = "./data/preprocessed_val"
    dataset_dict['train'].save_to_disk(train_path)
    dataset_dict['validation'].save_to_disk(val_path)
    print("Pretraining data saved.")
    
    # 3. Download and cache all downstream evaluation datasets
    eval_dir = "./data/eval_data"
    os.makedirs(eval_dir, exist_ok=True)
    print(f"\nCaching downstream evaluation datasets to {eval_dir} for offline compatibility...")
    
    # A. WikiText-2 (Perplexity)
    try:
        print("Caching WikiText-2...")
        wt = load_dataset("wikitext", "wikitext-2-raw-v1")
        wt.save_to_disk(os.path.join(eval_dir, "wikitext-2"))
    except Exception as e:
        print(f"Warning caching WikiText-2: {e}")
        
    # B. GLUE Tasks
    for task in ["mrpc", "cola", "sst2"]:
        try:
            print(f"Caching GLUE task: {task}...")
            gd = load_dataset("glue", task)
            gd.save_to_disk(os.path.join(eval_dir, f"glue_{task}"))
        except Exception as e:
            print(f"Warning caching GLUE {task}: {e}")
            
    # C. BLiMP Subsets
    for subset in ["regular_plural_subject_verb_agreement_1", "anaphor_gender_agreement"]:
        try:
            print(f"Caching BLiMP subset: {subset}...")
            bl = load_dataset("nyu-mll/blimp", subset)
            bl.save_to_disk(os.path.join(eval_dir, f"blimp_{subset}"))
        except Exception as e:
            print(f"Warning caching BLiMP {subset}: {e}")
            
    # D. Multi30k (Translation for BLEU)
    try:
        print("Caching Multi30k...")
        m30 = load_dataset("bentrevett/multi30k")
        m30.save_to_disk(os.path.join(eval_dir, "multi30k"))
    except Exception as e:
        print(f"Warning caching Multi30k: {e}")
        
    print("\nAll datasets preprocessed and cached successfully!")

if __name__ == "__main__":
    main()
