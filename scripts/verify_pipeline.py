import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertConfig

# Add project root to path so src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import PretrainDataset, MultiTaskCollator
from src.models import MultiTaskBERT

def run_tests():
    print("=== STARTING PIPELINE VERIFICATION TESTS ===")
    
    # 1. Initialize Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # 2. Create Mock Dataset (Sentences with POS labels)
    # POS tag mapping: let's assume 'NN': 0, 'VB': 1, 'JJ': 2, 'PRP': 3, 'IN': 4, '.': 5
    mock_pos_to_id = {'NN': 0, 'VB': 1, 'JJ': 2, 'PRP': 3, 'IN': 4, '.': 5}
    
    # Create 3 sentences
    # S1: "the cat sat on the mat ."
    # S2: "she likes happy cats ."
    # S3: "a quick brown fox jumps ."
    mock_data = [
        {
            "words": ["the", "cat", "sat", "on", "the", "mat", "."],
            "pos_labels": [2, 0, 1, 4, 2, 0, 5]
        },
        {
            "words": ["she", "likes", "happy", "cats", "."],
            "pos_labels": [3, 1, 2, 0, 5]
        },
        {
            "words": ["a", "quick", "brown", "fox", "jumps", "."],
            "pos_labels": [2, 2, 2, 0, 1, 5]
        }
    ]
    
    # --- TEST 1: PretrainDataset (NSP Active vs Inactive) ---
    print("\n--- Running Test 1: PretrainDataset ---")
    dataset_nsp = PretrainDataset(mock_data, nsp_active=True)
    dataset_no_nsp = PretrainDataset(mock_data, nsp_active=False)
    
    assert len(dataset_nsp) == 3
    assert len(dataset_no_nsp) == 3
    
    # Test item contents
    item_nsp = dataset_nsp[0]
    assert "words_a" in item_nsp
    assert "words_b" in item_nsp
    assert "pos_a" in item_nsp
    assert "pos_b" in item_nsp
    assert "next_label" in item_nsp
    
    if item_nsp["next_label"] == 0:
        # Consecutive
        assert item_nsp["words_b"] == mock_data[1]["words"]
    else:
        # Random
        assert item_nsp["words_b"] in [mock_data[0]["words"], mock_data[1]["words"], mock_data[2]["words"]]
        
    item_no_nsp = dataset_no_nsp[0]
    assert item_no_nsp["words_b"] is None
    assert item_no_nsp["next_label"] == 0
    print("Test 1 Passed: Dataset pairing and labels are correct!")

    # --- TEST 2: MultiTaskCollator (NSP Active) ---
    print("\n--- Running Test 2: MultiTaskCollator (NSP Active) ---")
    collator_nsp = MultiTaskCollator(tokenizer, mlm_active=True, mlm_probability=0.15)
    
    # Create batch
    batch_items = [dataset_nsp[0], dataset_nsp[1]]
    batch = collator_nsp(batch_items)
    
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "token_type_ids" in batch
    assert "labels" in batch
    assert "next_sentence_label" in batch
    assert "word_ids" in batch
    assert "pos_labels" in batch
    
    # Check dimensions
    batch_size, seq_len = batch["input_ids"].shape
    assert batch_size == 2
    assert batch["attention_mask"].shape == (batch_size, seq_len)
    assert batch["token_type_ids"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)
    assert batch["word_ids"].shape == (batch_size, seq_len)
    assert batch["pos_labels"].shape == (batch_size, seq_len)
    assert batch["next_sentence_label"].shape == (batch_size,)
    
    # Verify Word ID offsets and POS alignment
    # If NSP is active, token_type_ids has 1s for the second sentence
    # Word IDs should be offset for segment 1
    for b in range(batch_size):
        w_ids = batch["word_ids"][b].tolist()
        t_type = batch["token_type_ids"][b].tolist()
        input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][b])
        
        # Special tokens ([CLS]=101, [SEP]=102, [PAD]=0) should have word_id = -1 and pos_label = -100
        for idx, (w_id, t_t) in enumerate(zip(w_ids, t_type)):
            tok = input_tokens[idx]
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                assert w_id == -1, f"Special token {tok} got word_id {w_id}"
                assert batch["pos_labels"][b, idx].item() == -100
            else:
                assert w_id >= 0
                
    # Verify MLM masking: labels should only have active values (not -100) at masked indices
    masked_count = (batch["labels"] != -100).sum().item()
    print(f"MLM Masked tokens count: {masked_count} out of {batch_size * seq_len}")
    
    # Verify that special tokens are never masked
    for b in range(batch_size):
        input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][b])
        for idx, tok in enumerate(input_tokens):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                assert batch["labels"][b, idx].item() == -100, f"Special token {tok} was masked!"
                
    print("Test 2 Passed: Collator correctly aligned word IDs, POS labels, and applied MLM masking!")

    # --- TEST 3: MultiTaskBERT Model Forward Pass ---
    print("\n--- Running Test 3: MultiTaskBERT Model Forward Pass ---")
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    
    model = MultiTaskBERT(config, num_pos_tags=len(mock_pos_to_id))
    
    # Run forward pass for all tasks
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        labels=batch["labels"],
        next_sentence_label=batch["next_sentence_label"],
        word_ids=batch["word_ids"],
        pos_labels=batch["pos_labels"],
        tasks=["mlm", "nsp", "pos"],
        alpha=1.0
    )
    
    assert "loss" in outputs
    assert "mlm_loss" in outputs
    assert "nsp_loss" in outputs
    assert "pos_loss" in outputs
    assert "mlm_logits" in outputs
    assert "nsp_logits" in outputs
    assert "pos_logits" in outputs
    
    # Check loss bounds and types
    assert outputs["loss"].item() > 0.0
    assert outputs["mlm_loss"] > 0.0
    assert outputs["nsp_loss"] > 0.0
    assert outputs["pos_loss"] > 0.0
    
    # Verify output shapes of logits
    max_words = int(batch["word_ids"][batch["word_ids"] >= 0].max().item()) + 1
    assert outputs["mlm_logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs["nsp_logits"].shape == (batch_size, 2)
    assert outputs["pos_logits"].shape == (batch_size, max_words, len(mock_pos_to_id))
    
    print(f"Aggregated Loss: {outputs['loss'].item():.4f}")
    print(f"MLM Loss: {outputs['mlm_loss']:.4f}")
    print(f"NSP Loss: {outputs['nsp_loss']:.4f}")
    print(f"POS Loss: {outputs['pos_loss']:.4f}")
    
    # Test Backpropagation
    loss = outputs["loss"]
    loss.backward()
    
    # Check if gradients are populated and valid
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has inf gradient!"
            
    print("Test 3 Passed: Model forward pass, loss calculation, and backpropagation are fully functional and mathematically sound!")

    print("\n=== ALL PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
